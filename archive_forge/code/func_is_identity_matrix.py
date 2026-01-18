from __future__ import annotations
import numpy as np
def is_identity_matrix(mat, ignore_phase=False, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is an identity matrix."""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    if ignore_phase:
        theta = np.angle(mat[0, 0])
        mat = np.exp(-1j * theta) * mat
    iden = np.eye(len(mat))
    return np.allclose(mat, iden, rtol=rtol, atol=atol)