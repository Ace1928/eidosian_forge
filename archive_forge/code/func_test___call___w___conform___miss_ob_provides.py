import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w___conform___miss_ob_provides(self):
    ib = self._makeOne(True)

    class _Adapted:

        def __conform__(self, iface):
            return None
    adapted = _Adapted()
    self.assertIs(ib(adapted), adapted)