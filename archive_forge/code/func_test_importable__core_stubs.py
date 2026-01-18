from os import path
import pickle
import numpy as np
def test_importable__core_stubs(self):
    """
        Checks if stubs for `numpy._core` are importable.
        """
    from numpy._core.multiarray import _reconstruct
    from numpy._core.umath import cos
    from numpy._core._multiarray_umath import exp
    from numpy._core._internal import ndarray
    from numpy._core._dtype import _construction_repr
    from numpy._core._dtype_ctypes import dtype_from_ctypes_type