import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__module_provides_typical_use(self):
    from zope.interface.tests import dummy
    provides = dummy.__provides__
    self.assertEqual(repr(provides), "directlyProvides(sys.modules['zope.interface.tests.dummy'], IDummyModule)")