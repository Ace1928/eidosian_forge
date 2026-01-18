import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w___conform___ob_no_provides_wo_alternate(self):
    ib = self._makeOne(False)
    with self.assertRaises(TypeError) as exc:
        ib(object())
    self.assertIn('Could not adapt', str(exc.exception))