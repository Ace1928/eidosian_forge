import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___hash___missing_required_attrs(self):

    class Derived(self._getTargetClass()):

        def __init__(self):
            pass
    derived = Derived()
    with self.assertRaises(AttributeError):
        hash(derived)