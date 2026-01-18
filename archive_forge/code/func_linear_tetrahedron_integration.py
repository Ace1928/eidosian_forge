from math import pi, sqrt
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.parallel import world
from ase.utils.cext import cextension
def linear_tetrahedron_integration(cell, eigs, energies, weights=None, comm=world):
    """DOS from linear tetrahedron interpolation.

    cell: 3x3 ndarray-like
        Unit cell.
    eigs: (n1, n2, n3, nbands)-shaped ndarray
        Eigenvalues on a Monkhorst-Pack grid (not reduced).
    energies: 1-d array-like
        Energies where the DOS is calculated (must be a uniform grid).
    weights: ndarray of shape (n1, n2, n3, nbands) or (n1, n2, n3, nbands, nw)
        Weights.  Defaults to a (n1, n2, n3, nbands)-shaped ndarray
        filled with ones.  Can also have an extra dimednsion if there are
        nw weights.
    comm: communicator object
            MPI communicator for lti_dos

    Returns:

        DOS as an ndarray of same length as energies or as an
        ndarray of shape (nw, len(energies)).

    See:

        Extensions of the tetrahedron method for evaluating
        spectral properties of solids,
        A. H. MacDonald, S. H. Vosko and P. T. Coleridge,
        1979 J. Phys. C: Solid State Phys. 12 2991,
        https://doi.org/10.1088/0022-3719/12/15/008
    """
    from scipy.spatial import Delaunay
    size = eigs.shape[:3]
    B = (np.linalg.inv(cell) / size).T
    indices = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]])
    dt = Delaunay(np.dot(indices, B))
    if weights is None:
        weights = np.ones_like(eigs)
    if weights.ndim == 4:
        extra_dimension_added = True
        weights = weights[:, :, :, :, np.newaxis]
    else:
        extra_dimension_added = False
    nweights = weights.shape[4]
    dos = np.empty((nweights, len(energies)))
    lti_dos(indices[dt.simplices], eigs, weights, energies, dos, comm)
    dos /= np.prod(size)
    if extra_dimension_added:
        return dos[0]
    return dos