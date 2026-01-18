import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_gep_castinstr_addrspace(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    addrspace = 4
    int8ptr = int8.as_pointer()
    ls = ir.LiteralStructType([int64, int8ptr, int8ptr, int8ptr, int64])
    d = builder.bitcast(a, ls.as_pointer(addrspace=addrspace), name='d')
    e = builder.gep(d, [ir.Constant(int32, x) for x in [0, 3]], name='e')
    self.assertEqual(e.type.addrspace, addrspace)
    self.assertEqual(e.type, ir.PointerType(int8ptr, addrspace=addrspace))
    self.check_block(block, '            my_block:\n                %"d" = bitcast i32 %".1" to {i64, i8*, i8*, i8*, i64} addrspace(4)*\n                %"e" = getelementptr {i64, i8*, i8*, i8*, i64}, {i64, i8*, i8*, i8*, i64} addrspace(4)* %"d", i32 0, i32 3\n            ')