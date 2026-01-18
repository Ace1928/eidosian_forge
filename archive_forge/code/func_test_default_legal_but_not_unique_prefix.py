import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_default_legal_but_not_unique_prefix(self):

    class A(HasTraits):
        foo = PrefixList(['live', 'modal', 'livemodal'], default='live')
    a = A()
    self.assertEqual(a.foo, 'live')