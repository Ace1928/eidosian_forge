import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_binops_with_overflow(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    builder.sadd_with_overflow(a, b, 'c')
    builder.smul_with_overflow(a, b, 'd')
    builder.ssub_with_overflow(a, b, 'e')
    builder.uadd_with_overflow(a, b, 'f')
    builder.umul_with_overflow(a, b, 'g')
    builder.usub_with_overflow(a, b, 'h')
    self.check_block(block, 'my_block:\n    %"c" = call {i32, i1} @"llvm.sadd.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"d" = call {i32, i1} @"llvm.smul.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"e" = call {i32, i1} @"llvm.ssub.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"f" = call {i32, i1} @"llvm.uadd.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"g" = call {i32, i1} @"llvm.umul.with.overflow.i32"(i32 %".1", i32 %".2")\n    %"h" = call {i32, i1} @"llvm.usub.with.overflow.i32"(i32 %".1", i32 %".2")\n            ')