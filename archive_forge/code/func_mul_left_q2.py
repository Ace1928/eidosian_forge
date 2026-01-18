import numpy as np
from .layer import Layer1Q, Layer2Q
def mul_left_q2(self, layer: Layer2Q, temp_mat: np.ndarray):
    """
        Multiplies ``NxN`` matrix, wrapped by this object, by a 2-qubit layer
        matrix on the left, where ``dim`` is the actual size of matrices involved,
        ``dim = 2^{num. of qubits}``.

        Args:
            layer: 2-qubit layer, i.e. the layer with just one non-trivial
                   2-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
        """
    mat = self._mat
    gmat, perm, inv_perm = layer.get_attr()
    dim = perm.size
    np.take(mat, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)
    if dim > 512:
        for i in range(0, dim, 4):
            np.dot(gmat, temp_mat[i:i + 4, :], out=mat[i:i + 4, :])
    else:
        half = dim // 4
        np.copyto(mat.reshape((4, half, dim)), np.swapaxes(temp_mat.reshape((half, 4, dim)), 0, 1))
        np.dot(gmat, mat.reshape(4, -1), out=temp_mat.reshape(4, -1))
        np.copyto(mat.reshape((half, 4, dim)), np.swapaxes(temp_mat.reshape((4, half, dim)), 0, 1))
    self._left_perm[:] = inv_perm