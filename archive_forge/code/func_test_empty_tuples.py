from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def test_empty_tuples(self):
    fe_args = [types.UniTuple(types.int16, 0), types.Tuple(()), types.int32]
    self._test_as_arguments(fe_args)