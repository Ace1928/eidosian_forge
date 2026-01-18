import numpy as np
from ase.constraints import Filter, FixAtoms
from ase.geometry.cell import cell_to_cellpar
from ase.neighborlist import neighbor_list
def estimate_nearest_neighbour_distance(atoms, neighbor_list=neighbor_list):
    """
    Estimate nearest neighbour distance r_NN

    Args:
        atoms: Atoms object
        neighbor_list: function (optional). Optionally replace the built-in
            ASE neighbour list with an alternative with the same call
            signature, e.g. `matscipy.neighbours.neighbour_list`.        

    Returns:
        rNN: float
            Nearest neighbour distance
    """
    if isinstance(atoms, Filter):
        atoms = atoms.atoms
    r_cut = 1.0
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
    extent = [a, b, c]
    while r_cut < 2.0 * max(extent):
        i, j, rij, fixed_atoms = get_neighbours(atoms, r_cut, self_interaction=True, neighbor_list=neighbor_list)
        if len(i) != 0:
            nn_i = np.bincount(i, minlength=len(atoms))
            if (nn_i != 0).all():
                break
        r_cut *= phi
    else:
        raise RuntimeError('increased r_cut to twice system extent without finding neighbours for all atoms. This can happen if your system is too small; try setting r_cut manually')
    nn_distances = [np.min(rij[i == I]) for I in range(len(atoms))]
    r_NN = np.max(nn_distances)
    return r_NN