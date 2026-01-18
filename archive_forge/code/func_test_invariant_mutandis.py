import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_invariant_mutandis(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface import directlyProvides
    from zope.interface import invariant

    class IInvariant(Interface):
        foo = Attribute('foo')
        bar = Attribute('bar; must eval to Boolean True if foo does')
        invariant(_ifFooThenBar)

    class HasInvariant:
        pass
    has_invariant = HasInvariant()
    directlyProvides(has_invariant, IInvariant)
    has_invariant.foo = 42
    old_invariants = IInvariant.getTaggedValue('invariants')
    invariants = old_invariants[:]
    invariants.append(_barGreaterThanFoo)
    IInvariant.setTaggedValue('invariants', invariants)
    self._errorsEqual(has_invariant, 1, ['If Foo, then Bar!'], IInvariant)
    has_invariant.foo = 2
    has_invariant.bar = 1
    self._errorsEqual(has_invariant, 1, ['Please, Boo MUST be greater than Foo!'], IInvariant)
    has_invariant.foo = 1
    has_invariant.bar = 0
    self._errorsEqual(has_invariant, 2, ['If Foo, then Bar!', 'Please, Boo MUST be greater than Foo!'], IInvariant)
    has_invariant.foo = 1
    has_invariant.bar = 2
    self.assertEqual(IInvariant.validateInvariants(has_invariant), None)
    IInvariant.setTaggedValue('invariants', old_invariants)