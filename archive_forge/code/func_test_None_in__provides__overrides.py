import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_None_in__provides__overrides(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class IFoo(Interface):
        pass

    @implementer(IFoo)
    class Foo:

        @property
        def __provides__(self):
            return None
    Foo.__providedBy__ = self._makeOne()
    provided = getattr(Foo(), '__providedBy__')
    self.assertIsNone(provided)