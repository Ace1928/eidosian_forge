import numpy as np
from ase import Atoms
def make_supercell(prim, P, wrap=True, tol=1e-05):
    """Generate a supercell by applying a general transformation (*P*) to
    the input configuration (*prim*).

    The transformation is described by a 3x3 integer matrix
    `\\mathbf{P}`. Specifically, the new cell metric
    `\\mathbf{h}` is given in terms of the metric of the input
    configuration `\\mathbf{h}_p` by `\\mathbf{P h}_p =
    \\mathbf{h}`.

    Parameters:

    prim: ASE Atoms object
        Input configuration.
    P: 3x3 integer matrix
        Transformation matrix `\\mathbf{P}`.
    wrap: bool
        wrap in the end
    tol: float
        tolerance for wrapping
    """
    supercell_matrix = P
    supercell = clean_matrix(supercell_matrix @ prim.cell)
    lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell)
    superatoms = Atoms(cell=supercell, pbc=prim.pbc)
    for lp in lattice_points:
        shifted_atoms = prim.copy()
        shifted_atoms.positions += lp
        superatoms.extend(shifted_atoms)
    n_target = int(np.round(np.linalg.det(supercell_matrix) * len(prim)))
    if n_target != len(superatoms):
        msg = 'Number of atoms in supercell: {}, expected: {}'.format(n_target, len(superatoms))
        raise SupercellError(msg)
    if wrap:
        superatoms.wrap(eps=tol)
    return superatoms