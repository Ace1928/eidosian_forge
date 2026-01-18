import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_ctlz(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int16, 5)
    b = ir.Constant(int1, 1)
    c = builder.ctlz(a, b, name='c')
    builder.ret(c)
    self.check_block(block, '            my_block:\n                %"c" = call i16 @"llvm.ctlz.i16"(i16 5, i1 1)\n                ret i16 %"c"\n            ')