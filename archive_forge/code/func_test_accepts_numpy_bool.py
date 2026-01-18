import unittest
from traits.api import Bool, Dict, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
@requires_numpy
def test_accepts_numpy_bool(self):
    a = A()
    a.foo = numpy.bool_(True)
    self.assertTrue(a.foo)