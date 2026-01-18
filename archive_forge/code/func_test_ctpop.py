import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_ctpop(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int16, 5)
    c = builder.ctpop(a, name='c')
    builder.ret(c)
    self.check_block(block, '            my_block:\n                %"c" = call i16 @"llvm.ctpop.i16"(i16 5)\n                ret i16 %"c"\n            ')