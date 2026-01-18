import operator
import numpy as np
from numpy import ndarray, float_
import numpy.core.umath as umath
import numpy.testing
from numpy.testing import (
from .core import mask_or, getmask, masked_array, nomask, masked, filled
from unittest import TestCase
def fail_if_equal(actual, desired, err_msg=''):
    """
    Raises an assertion error if two items are equal.

    """
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        fail_if_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            fail_if_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        fail_if_equal(len(actual), len(desired), err_msg)
        for k in range(len(desired)):
            fail_if_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')
        return
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return fail_if_array_equal(actual, desired, err_msg)
    msg = build_err_msg([actual, desired], err_msg)
    if not desired != actual:
        raise AssertionError(msg)