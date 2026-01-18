import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def primitive_neighbor_list(quantities, pbc, cell, positions, cutoff, numbers=None, self_interaction=False, use_scaled_positions=False, max_nbins=1000000.0):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters:

    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are

            * 'i' : first atom index
            * 'j' : second atom index
            * 'd' : absolute distance
            * 'D' : distance vector
            * 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j). With the shift vector S, the
              distances D between atoms can be computed from:
              D = positions[j]-positions[i]+S.dot(cell)
    pbc: array_like
        3-tuple indicating giving periodic boundaries in the three Cartesian
        directions.
    cell: 3x3 matrix
        Unit cell vectors.
    positions: list of xyz-positions
        Atomic positions.  Anything that can be converted to an ndarray of
        shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
        use_scaled_positions is set to true, this must be scaled positions.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    use_scaled_positions: bool
        If set to true, positions are expected to be scaled positions.
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns:

    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a)-1, but the order of (i,j)
        pairs is not guaranteed.

    """
    if len(positions) == 0:
        empty_types = dict(i=(int, (0,)), j=(int, (0,)), D=(float, (0, 3)), d=(float, (0,)), S=(int, (0, 3)))
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [np.array([], dtype=dtype).reshape(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)
    b1_c, b2_c, b3_c = np.linalg.pinv(cell).T
    l1 = np.linalg.norm(b1_c)
    l2 = np.linalg.norm(b2_c)
    l3 = np.linalg.norm(b3_c)
    face_dist_c = np.array([1 / l1 if l1 > 0 else 1, 1 / l2 if l2 > 0 else 1, 1 / l3 if l3 > 0 else 1])
    if isinstance(cutoff, dict):
        max_cutoff = max(cutoff.values())
    elif np.isscalar(cutoff):
        max_cutoff = cutoff
    else:
        cutoff = np.asarray(cutoff)
        max_cutoff = 2 * np.max(cutoff)
    bin_size = max(max_cutoff, 3)
    nbins_c = np.maximum((face_dist_c / bin_size).astype(int), [1, 1, 1])
    nbins = np.prod(nbins_c)
    while nbins > max_nbins:
        nbins_c = np.maximum(nbins_c // 2, [1, 1, 1])
        nbins = np.prod(nbins_c)
    neigh_search_x, neigh_search_y, neigh_search_z = np.ceil(bin_size * nbins_c / face_dist_c).astype(int)
    neigh_search_x = 0 if nbins_c[0] == 1 and (not pbc[0]) else neigh_search_x
    neigh_search_y = 0 if nbins_c[1] == 1 and (not pbc[1]) else neigh_search_y
    neigh_search_z = 0 if nbins_c[2] == 1 and (not pbc[2]) else neigh_search_z
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = np.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = np.linalg.solve(complete_cell(cell).T, positions.T).T
    bin_index_ic = np.floor(scaled_positions_ic * nbins_c).astype(int)
    cell_shift_ic = np.zeros_like(bin_index_ic)
    for c in range(3):
        if pbc[c]:
            cell_shift_ic[:, c], bin_index_ic[:, c] = divmod(bin_index_ic[:, c], nbins_c[c])
        else:
            bin_index_ic[:, c] = np.clip(bin_index_ic[:, c], 0, nbins_c[c] - 1)
    bin_index_i = bin_index_ic[:, 0] + nbins_c[0] * (bin_index_ic[:, 1] + nbins_c[1] * bin_index_ic[:, 2])
    atom_i = np.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]
    max_natoms_per_bin = np.bincount(bin_index_i).max()
    atoms_in_bin_ba = -np.ones([nbins, max_natoms_per_bin], dtype=int)
    for i in range(max_natoms_per_bin):
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]
        mask = np.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0
    atom_pairs_pn = np.indices((max_natoms_per_bin, max_natoms_per_bin), dtype=int)
    atom_pairs_pn = atom_pairs_pn.reshape(2, -1)
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []
    binz_xyz, biny_xyz, binx_xyz = np.meshgrid(np.arange(nbins_c[2]), np.arange(nbins_c[1]), np.arange(nbins_c[0]), indexing='ij')
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-neigh_search_z, neigh_search_z + 1):
        for dy in range(-neigh_search_y, neigh_search_y + 1):
            for dx in range(-neigh_search_x, neigh_search_x + 1):
                shiftx_xyz, neighbinx_xyz = divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (neighbinx_xyz + nbins_c[0] * (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)).ravel()
                _secnd_at_neightuple_n = atoms_in_bin_ba[neighbin_b][:, atom_pairs_pn[1]]
                _cell_shift_vector_x_n = np.resize(shiftx_xyz.reshape(-1, 1), (max_natoms_per_bin ** 2, shiftx_xyz.size)).T
                _cell_shift_vector_y_n = np.resize(shifty_xyz.reshape(-1, 1), (max_natoms_per_bin ** 2, shifty_xyz.size)).T
                _cell_shift_vector_z_n = np.resize(shiftz_xyz.reshape(-1, 1), (max_natoms_per_bin ** 2, shiftz_xyz.size)).T
                mask = np.logical_and(_first_at_neightuple_n != -1, _secnd_at_neightuple_n != -1)
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]
    first_at_neightuple_n = np.concatenate(first_at_neightuple_nn)
    secnd_at_neightuple_n = np.concatenate(secnd_at_neightuple_nn)
    cell_shift_vector_n = np.transpose([np.concatenate(cell_shift_vector_x_n), np.concatenate(cell_shift_vector_y_n), np.concatenate(cell_shift_vector_z_n)])
    cell_shift_vector_n += cell_shift_ic[first_at_neightuple_n] - cell_shift_ic[secnd_at_neightuple_n]
    if not self_interaction:
        m = np.logical_not(np.logical_and(first_at_neightuple_n == secnd_at_neightuple_n, (cell_shift_vector_n == 0).all(axis=1)))
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]
    i = np.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]
    distance_vector_nc = positions[secnd_at_neightuple_n] - positions[first_at_neightuple_n] + cell_shift_vector_n.dot(cell)
    abs_distance_vector_n = np.sqrt(np.sum(distance_vector_nc * distance_vector_nc, axis=1))
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]
    if isinstance(cutoff, dict) and numbers is not None:
        per_pair_cutoff_n = np.zeros_like(abs_distance_vector_n)
        for (atomic_number1, atomic_number2), c in cutoff.items():
            try:
                atomic_number1 = atomic_numbers[atomic_number1]
            except KeyError:
                pass
            try:
                atomic_number2 = atomic_numbers[atomic_number2]
            except KeyError:
                pass
            if atomic_number1 == atomic_number2:
                mask = np.logical_and(numbers[first_at_neightuple_n] == atomic_number1, numbers[secnd_at_neightuple_n] == atomic_number2)
            else:
                mask = np.logical_or(np.logical_and(numbers[first_at_neightuple_n] == atomic_number1, numbers[secnd_at_neightuple_n] == atomic_number2), np.logical_and(numbers[first_at_neightuple_n] == atomic_number2, numbers[secnd_at_neightuple_n] == atomic_number1))
            per_pair_cutoff_n[mask] = c
        mask = abs_distance_vector_n < per_pair_cutoff_n
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    elif not np.isscalar(cutoff):
        mask = abs_distance_vector_n < cutoff[first_at_neightuple_n] + cutoff[secnd_at_neightuple_n]
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    retvals = []
    for q in quantities:
        if q == 'i':
            retvals += [first_at_neightuple_n]
        elif q == 'j':
            retvals += [secnd_at_neightuple_n]
        elif q == 'D':
            retvals += [distance_vector_nc]
        elif q == 'd':
            retvals += [abs_distance_vector_n]
        elif q == 'S':
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError('Unsupported quantity specified.')
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)