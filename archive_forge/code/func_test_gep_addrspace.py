import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_gep_addrspace(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    addrspace = 4
    c = builder.alloca(ir.PointerType(int32, addrspace=addrspace), name='c')
    self.assertEqual(str(c.type), 'i32 addrspace(4)**')
    self.assertEqual(c.type.pointee.addrspace, addrspace)
    d = builder.gep(c, [ir.Constant(int32, 5), a], name='d')
    self.assertEqual(d.type.addrspace, addrspace)
    e = builder.gep(d, [ir.Constant(int32, 10)], name='e')
    self.assertEqual(e.type.addrspace, addrspace)
    self.check_block(block, '            my_block:\n                %"c" = alloca i32 addrspace(4)*\n                %"d" = getelementptr i32 addrspace(4)*, i32 addrspace(4)** %"c", i32 5, i32 %".1"\n                %"e" = getelementptr i32, i32 addrspace(4)* %"d", i32 10\n            ')