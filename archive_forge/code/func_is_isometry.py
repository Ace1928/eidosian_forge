from __future__ import annotations
import numpy as np
def is_isometry(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is an isometry."""
    mat = np.array(mat)
    iden = np.eye(mat.shape[1])
    mat = np.conj(mat.T).dot(mat)
    return np.allclose(mat, iden, rtol=rtol, atol=atol)