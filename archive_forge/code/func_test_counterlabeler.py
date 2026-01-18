import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_counterlabeler(self):
    m = self.m
    lbl = CounterLabeler()
    self.assertEqual(lbl(m.mycomp), 1)
    self.assertEqual(lbl(m.mycomp), 2)
    self.assertEqual(lbl(m.that), 3)
    self.assertEqual(lbl(self.long1), 4)
    self.assertEqual(lbl(m.myblock), 5)
    self.assertEqual(lbl(m.myblock.mystreet), 6)
    self.assertEqual(lbl(self.thecopy), 7)