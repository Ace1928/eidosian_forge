import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_misc_ops(self):
    block = self.block(name='my_block')
    t = ir.Constant(int1, True)
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    builder.select(t, a, b, 'c', flags=('arcp', 'nnan'))
    self.assertFalse(block.is_terminated)
    builder.unreachable()
    self.assertTrue(block.is_terminated)
    self.check_block(block, '            my_block:\n                %"c" = select arcp nnan i1 true, i32 %".1", i32 %".2"\n                unreachable\n            ')