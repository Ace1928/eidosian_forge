import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_gep_addrspace_globalvar(self):
    m = self.module()
    tp = ir.LiteralStructType((flt, int1))
    addrspace = 4
    gv = ir.GlobalVariable(m, tp, 'myconstant', addrspace=addrspace)
    self.assertEqual(gv.addrspace, addrspace)
    c = gv.gep([ir.Constant(int32, x) for x in (0, 1)])
    self.assertEqual(c.type.addrspace, addrspace)
    self.assertEqual(str(c), 'getelementptr ({float, i1}, {float, i1} addrspace(4)* @"myconstant", i32 0, i32 1)')
    self.assertEqual(c.type, ir.PointerType(int1, addrspace=addrspace))