from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def callback_python(a, user_data=None):
    if a == ERROR_VALUE:
        raise ValueError('bad value')
    if user_data is None:
        return a + 1
    else:
        return a + user_data