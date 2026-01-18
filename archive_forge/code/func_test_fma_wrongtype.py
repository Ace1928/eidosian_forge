import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_fma_wrongtype(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int32, 5)
    b = ir.Constant(int32, 1)
    c = ir.Constant(int32, 2)
    with self.assertRaises(TypeError) as raises:
        builder.fma(a, b, c, name='fma')
    self.assertIn('expected an floating point type, got i32', str(raises.exception))