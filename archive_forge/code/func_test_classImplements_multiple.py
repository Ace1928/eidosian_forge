import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_classImplements_multiple(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import providedBy

    class ILeft(Interface):

        def method():
            """docstring"""

    class IRight(ILeft):
        pass

    class Left:
        __implemented__ = ILeft

        def method(self):
            raise NotImplementedError()

    class Right:
        __implemented__ = IRight

    class Ambi(Left, Right):
        pass
    ambi = Ambi()
    self.assertTrue(ILeft.implementedBy(Ambi))
    self.assertTrue(IRight.implementedBy(Ambi))
    self.assertTrue(ILeft in implementedBy(Ambi))
    self.assertTrue(IRight in implementedBy(Ambi))
    self.assertTrue(ILeft in providedBy(ambi))
    self.assertTrue(IRight in providedBy(ambi))