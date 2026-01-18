from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def test_issue2921(self):
    import numpy as np
    from numba import njit

    @njit
    def copy(a, b):
        for i in range(a.shape[0]):
            a[i] = b[i]
    b = np.arange(5, dtype=np.uint8).view(np.bool_)
    a = np.zeros_like(b)
    copy(a, b)
    np.testing.assert_equal(a, np.array((False,) + (True,) * 4))