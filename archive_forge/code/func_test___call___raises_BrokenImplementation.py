import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___raises_BrokenImplementation(self):
    from zope.interface.exceptions import BrokenImplementation
    method = self._makeOne()
    try:
        method()
    except BrokenImplementation as e:
        self.assertEqual(e.interface, None)
        self.assertEqual(e.name, self.DEFAULT_NAME)
    else:
        self.fail('__call__ should raise BrokenImplementation')