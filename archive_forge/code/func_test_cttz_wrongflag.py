import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_cttz_wrongflag(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int64, 5)
    b = ir.Constant(int32, 3)
    with self.assertRaises(TypeError) as raises:
        builder.cttz(a, b, name='c')
    self.assertIn('expected an i1 type, got i32', str(raises.exception))