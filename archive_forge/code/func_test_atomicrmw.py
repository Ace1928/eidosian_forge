import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_atomicrmw(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    c = builder.alloca(int32, name='c')
    d = builder.atomic_rmw('add', c, a, 'monotonic', 'd')
    self.assertEqual(d.type, int32)
    self.check_block(block, '            my_block:\n                %"c" = alloca i32\n                %"d" = atomicrmw add i32* %"c", i32 %".1" monotonic\n            ')