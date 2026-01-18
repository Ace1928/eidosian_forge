import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_invariant_nested(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import directlyProvides
    from zope.interface import invariant

    class IInvariant(Interface):
        foo = Attribute('foo')
        bar = Attribute('bar; must eval to Boolean True if foo does')
        invariant(_ifFooThenBar)

    class ISubInvariant(IInvariant):
        invariant(_barGreaterThanFoo)

    class HasInvariant:
        pass
    self.assertEqual(ISubInvariant.getTaggedValue('invariants'), [_barGreaterThanFoo])
    has_invariant = HasInvariant()
    directlyProvides(has_invariant, ISubInvariant)
    has_invariant.foo = 42
    self._errorsEqual(has_invariant, 1, ['If Foo, then Bar!'], ISubInvariant)
    has_invariant.foo = 2
    has_invariant.bar = 1
    self._errorsEqual(has_invariant, 1, ['Please, Boo MUST be greater than Foo!'], ISubInvariant)
    has_invariant.foo = 1
    has_invariant.bar = 0
    self._errorsEqual(has_invariant, 2, ['If Foo, then Bar!', 'Please, Boo MUST be greater than Foo!'], ISubInvariant)
    has_invariant.foo = 1
    has_invariant.bar = 2
    self.assertEqual(IInvariant.validateInvariants(has_invariant), None)