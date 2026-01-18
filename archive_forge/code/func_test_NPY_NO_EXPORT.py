import sys
import sysconfig
import subprocess
import pkgutil
import types
import importlib
import warnings
import numpy as np
import numpy
import pytest
from numpy.testing import IS_WASM
@pytest.mark.skipif(ctypes is None, reason='ctypes not available in this python')
def test_NPY_NO_EXPORT():
    cdll = ctypes.CDLL(np.core._multiarray_tests.__file__)
    f = getattr(cdll, 'test_not_exported', None)
    assert f is None, "'test_not_exported' is mistakenly exported, NPY_NO_EXPORT does not work"