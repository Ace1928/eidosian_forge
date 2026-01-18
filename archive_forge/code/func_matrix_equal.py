from __future__ import annotations
import numpy as np
def matrix_equal(mat1, mat2, ignore_phase=False, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT, props=None):
    "Test if two arrays are equal.\n\n    The final comparison is implemented using Numpy.allclose. See its\n    documentation for additional information on tolerance parameters.\n\n    If ignore_phase is True both matrices will be multiplied by\n    exp(-1j * theta) where `theta` is the first nphase for a\n    first non-zero matrix element `|a| * exp(1j * theta)`.\n\n    Args:\n        mat1 (matrix_like): a matrix\n        mat2 (matrix_like): a matrix\n        ignore_phase (bool): ignore complex-phase differences between\n            matrices [Default: False]\n        rtol (double): the relative tolerance parameter [Default {}].\n        atol (double): the absolute tolerance parameter [Default {}].\n        props (dict | None): if not ``None`` and ``ignore_phase`` is ``True``\n            returns the phase difference between the two matrices under\n            ``props['phase_difference']``\n\n    Returns:\n        bool: True if the matrices are equal or False otherwise.\n    ".format(RTOL_DEFAULT, ATOL_DEFAULT)
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    if not isinstance(mat1, np.ndarray):
        mat1 = np.array(mat1)
    if not isinstance(mat2, np.ndarray):
        mat2 = np.array(mat2)
    if mat1.shape != mat2.shape:
        return False
    if ignore_phase:
        phase_difference = 0
        for elt in mat1.flat:
            if abs(elt) > atol:
                angle = np.angle(elt)
                phase_difference -= angle
                mat1 = np.exp(-1j * angle) * mat1
                break
        for elt in mat2.flat:
            if abs(elt) > atol:
                angle = np.angle(elt)
                phase_difference += angle
                mat2 = np.exp(-1j * np.angle(elt)) * mat2
                break
        if props is not None:
            props['phase_difference'] = phase_difference
    return np.allclose(mat1, mat2, rtol=rtol, atol=atol)