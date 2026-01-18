from __future__ import annotations
import numpy as np
def is_hermitian_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a Hermitian matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, np.conj(mat.T), rtol=rtol, atol=atol)