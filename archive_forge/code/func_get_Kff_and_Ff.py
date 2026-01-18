import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def get_Kff_and_Ff(self, J, ecc, triangles, Uc):
    """
        Build K and F for the following elliptic formulation:
        minimization of curvature energy with value of function at node
        imposed and derivatives 'free'.

        Build the global Kff matrix in cco format.
        Build the full Ff vec Ff = - Kfc x Uc.

        Parameters
        ----------
        *J* is a (N x 2 x 2) array of jacobian matrices (jacobian matrix at
        triangle first apex)
        *ecc* is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities
        *triangles* is a (N x 3) array of nodes indexes.
        *Uc* is (N x 3) array of imposed displacements at nodes

        Returns
        -------
        (Kff_rows, Kff_cols, Kff_vals) Kff matrix in coo format - Duplicate
        (row, col) entries must be summed.
        Ff: force vector - dim npts * 3
        """
    ntri = np.size(ecc, 0)
    vec_range = np.arange(ntri, dtype=np.int32)
    c_indices = np.full(ntri, -1, dtype=np.int32)
    f_dof = [1, 2, 4, 5, 7, 8]
    c_dof = [0, 3, 6]
    f_dof_indices = _to_matrix_vectorized([[c_indices, triangles[:, 0] * 2, triangles[:, 0] * 2 + 1, c_indices, triangles[:, 1] * 2, triangles[:, 1] * 2 + 1, c_indices, triangles[:, 2] * 2, triangles[:, 2] * 2 + 1]])
    expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
    f_row_indices = _transpose_vectorized(expand_indices @ f_dof_indices)
    f_col_indices = expand_indices @ f_dof_indices
    K_elem = self.get_bending_matrices(J, ecc)
    Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
    Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
    Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])
    Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
    Uc_elem = np.expand_dims(Uc, axis=2)
    Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
    Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]
    Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
    return (Kff_rows, Kff_cols, Kff_vals, Ff)