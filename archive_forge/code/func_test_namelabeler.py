import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_namelabeler(self):
    m = self.m
    lbl = NameLabeler()
    self.assertEqual(lbl(m.mycomp), 'mycomp')
    self.assertEqual(lbl(m.mycomp), 'mycomp')
    self.assertEqual(lbl(m.that), 'that')
    self.assertEqual(lbl(self.long1), 'myverylongcomponentname')
    self.assertEqual(lbl(m.myblock), 'myblock')
    self.assertEqual(lbl(m.myblock.mystreet), 'myblock.mystreet')
    self.assertEqual(lbl(self.thecopy), "'myblock.mystreet'")
    self.assertEqual(lbl(m.ind[3]), 'ind[3]')
    self.assertEqual(lbl(m.ind[10]), 'ind[10]')
    self.assertEqual(lbl(m.ind[1]), 'ind[1]')