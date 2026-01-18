import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_cuidlabeler(self):
    m = self.m
    lbl = CuidLabeler()
    self.assertEqual(lbl(m.mycomp), ComponentUID(m.mycomp))
    self.assertEqual(lbl(m.mycomp), ComponentUID(m.mycomp))
    self.assertEqual(lbl(m.that), ComponentUID(m.that))
    self.assertEqual(lbl(self.long1), ComponentUID(self.long1))
    self.assertEqual(lbl(m.myblock), ComponentUID(m.myblock))
    self.assertEqual(lbl(m.myblock.mystreet), ComponentUID(m.myblock.mystreet))
    self.assertEqual(lbl(self.thecopy), ComponentUID(self.thecopy))
    self.assertEqual(lbl(m.ind[3]), ComponentUID(m.ind[3]))
    self.assertEqual(lbl(m.ind[10]), ComponentUID(m.ind[10]))
    self.assertEqual(lbl(m.ind[1]), ComponentUID(m.ind[1]))