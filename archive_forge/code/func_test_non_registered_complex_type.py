import re
import unittest
from wsme import exc
from wsme import types
def test_non_registered_complex_type(self):

    class TempType(types.Base):
        __registry__ = None
    self.assertFalse(types.iscomplex(TempType))
    types.registry.register(TempType)
    self.assertTrue(types.iscomplex(TempType))