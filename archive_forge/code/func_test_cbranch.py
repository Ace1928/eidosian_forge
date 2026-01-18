import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_cbranch(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    bb_true = builder.function.append_basic_block(name='b_true')
    bb_false = builder.function.append_basic_block(name='b_false')
    builder.cbranch(ir.Constant(int1, False), bb_true, bb_false)
    self.assertTrue(block.is_terminated)
    self.check_block(block, '            my_block:\n                br i1 false, label %"b_true", label %"b_false"\n            ')