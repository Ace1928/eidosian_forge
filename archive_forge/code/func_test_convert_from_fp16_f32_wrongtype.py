import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_convert_from_fp16_f32_wrongtype(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(flt, 5.5)
    with self.assertRaises(TypeError) as raises:
        builder.convert_from_fp16(a, name='b', to=flt)
    self.assertIn('expected an i16 type, got float', str(raises.exception))