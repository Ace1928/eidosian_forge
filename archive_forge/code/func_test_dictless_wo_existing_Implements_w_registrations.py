import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_dictless_wo_existing_Implements_w_registrations(self):
    from zope.interface import declarations

    class Foo:
        __slots__ = ('__implemented__',)
    foo = Foo()
    foo.__implemented__ = None
    reg = object()
    with _MonkeyDict(declarations, 'BuiltinImplementationSpecifications') as specs:
        specs[foo] = reg
        self.assertTrue(self._callFUT(foo) is reg)