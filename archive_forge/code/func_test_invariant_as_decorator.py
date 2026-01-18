import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_invariant_as_decorator(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface import invariant
    from zope.interface.exceptions import Invalid

    class IRange(Interface):
        min = Attribute('Lower bound')
        max = Attribute('Upper bound')

        @invariant
        def range_invariant(ob):
            if ob.max < ob.min:
                raise Invalid('max < min')

    @implementer(IRange)
    class Range:

        def __init__(self, min, max):
            self.min, self.max = (min, max)
    IRange.validateInvariants(Range(1, 2))
    IRange.validateInvariants(Range(1, 1))
    try:
        IRange.validateInvariants(Range(2, 1))
    except Invalid as e:
        self.assertEqual(str(e), 'max < min')