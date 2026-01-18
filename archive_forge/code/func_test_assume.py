import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_assume(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    c = builder.icmp_signed('>', a, b, name='c')
    builder.assume(c)
    self.check_block(block, '            my_block:\n                %"c" = icmp sgt i32 %".1", %".2"\n                call void @"llvm.assume"(i1 %"c")\n            ')