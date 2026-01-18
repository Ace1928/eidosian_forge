import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_convert_to_fp16_f32(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(flt, 5.0)
    b = builder.convert_to_fp16(a, name='b')
    builder.ret(b)
    self.check_block(block, '            my_block:\n                %"b" = call i16 @"llvm.convert.to.fp16.f32"(float 0x4014000000000000)\n                ret i16 %"b"\n            ')