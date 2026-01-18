import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w___conform___returning_value(self):
    ib = self._makeOne(False)
    conformed = object()

    class _Adapted:

        def __conform__(self, iface):
            return conformed
    self.assertIs(ib(_Adapted()), conformed)