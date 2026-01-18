import numpy as np
from .layer import Layer1Q, Layer2Q
def mul_right_q2(self, layer: Layer2Q, temp_mat: np.ndarray, dagger: bool=True):
    """
        Multiplies ``NxN`` matrix, wrapped by this object, by a 2-qubit layer
        matrix on the right, where ``N`` is the actual size of matrices involved,
        ``N = 2^{num. of qubits}``.

        Args:
            layer: 2-qubit layer, i.e. the layer with just one non-trivial
                   2-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
            dagger: if true, the right-hand side matrix will be taken as
                    conjugate transposed.
        """
    gmat, perm, inv_perm = layer.get_attr()
    mat = self._mat
    dim = perm.size
    np.take(mat, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)
    gmat_right = np.conj(gmat, out=self._temp_g4x4).T if dagger else gmat
    for i in range(0, dim, 4):
        mat[:, i:i + 4] = np.dot(temp_mat[:, i:i + 4], gmat_right, out=self._temp_slice_dim_x_4)
    self._right_perm[:] = inv_perm