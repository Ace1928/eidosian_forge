import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_bitreverse_wrongtype(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(flt, 5)
    with self.assertRaises(TypeError) as raises:
        builder.bitreverse(a, name='c')
    self.assertIn('expected an integer type, got float', str(raises.exception))