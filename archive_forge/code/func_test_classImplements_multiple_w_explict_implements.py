import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_classImplements_multiple_w_explict_implements(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import providedBy

    class ILeft(Interface):

        def method():
            """docstring"""

    class IRight(ILeft):
        pass

    class IOther(Interface):
        pass

    class Left:
        __implemented__ = ILeft

        def method(self):
            raise NotImplementedError()

    class Right:
        __implemented__ = IRight

    class Other:
        __implemented__ = IOther

    class Mixed(Left, Right):
        __implemented__ = (Left.__implemented__, Other.__implemented__)
    mixed = Mixed()
    self.assertTrue(ILeft.implementedBy(Mixed))
    self.assertFalse(IRight.implementedBy(Mixed))
    self.assertTrue(IOther.implementedBy(Mixed))
    self.assertTrue(ILeft in implementedBy(Mixed))
    self.assertFalse(IRight in implementedBy(Mixed))
    self.assertTrue(IOther in implementedBy(Mixed))
    self.assertTrue(ILeft in providedBy(mixed))
    self.assertFalse(IRight in providedBy(mixed))
    self.assertTrue(IOther in providedBy(mixed))