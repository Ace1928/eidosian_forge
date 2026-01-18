import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_instruction_removal(self):
    func = self.function()
    builder = ir.IRBuilder()
    blk = func.append_basic_block(name='entry')
    builder.position_at_end(blk)
    k = ir.Constant(int32, 1234)
    a = builder.add(k, k, 'a')
    retvoid = builder.ret_void()
    self.assertTrue(blk.is_terminated)
    builder.remove(retvoid)
    self.assertFalse(blk.is_terminated)
    b = builder.mul(a, a, 'b')
    c = builder.add(b, b, 'c')
    builder.remove(c)
    builder.ret_void()
    self.assertTrue(blk.is_terminated)
    self.check_block(blk, '            entry:\n                %"a" = add i32 1234, 1234\n                %"b" = mul i32 %"a", %"a"\n                ret void\n        ')