from __future__ import annotations
import numpy as np
def is_unitary_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a unitary matrix."""
    mat = np.array(mat)
    mat = np.conj(mat.T).dot(mat)
    return is_identity_matrix(mat, ignore_phase=False, rtol=rtol, atol=atol)