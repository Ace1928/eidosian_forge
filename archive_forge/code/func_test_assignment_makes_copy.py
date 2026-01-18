import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_assignment_makes_copy(self):
    f = Foo(l=['initial', 'value'])
    l = ['new']
    f.l = l
    self.assertEqual(l, f.l)
    self.assertIsNot(l, f.l)
    l.append('l change')
    self.assertNotIn('l change', f.l)
    f.l.append('f.l change')
    self.assertNotIn('f.l change', l)