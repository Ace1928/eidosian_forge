from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def test_int32_array_complex(self):
    fe_args = [types.int32, types.Array(types.int32, 1, 'C'), types.complex64]
    self._test_as_arguments(fe_args)