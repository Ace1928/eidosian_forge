import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__module_after_pickle(self):
    import pickle
    from zope.interface.tests import dummy
    provides = dummy.__provides__
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(provides, proto)