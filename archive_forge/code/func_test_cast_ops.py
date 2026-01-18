import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_cast_ops(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b, fa, ptr = builder.function.args[:4]
    c = builder.trunc(a, int8, name='c')
    d = builder.zext(c, int32, name='d')
    e = builder.sext(c, int32, name='e')
    fb = builder.fptrunc(fa, flt, 'fb')
    fc = builder.fpext(fb, dbl, 'fc')
    g = builder.fptoui(fa, int32, 'g')
    h = builder.fptosi(fa, int8, 'h')
    fd = builder.uitofp(g, flt, 'fd')
    fe = builder.sitofp(h, dbl, 'fe')
    i = builder.ptrtoint(ptr, int32, 'i')
    j = builder.inttoptr(i, ir.PointerType(int8), 'j')
    k = builder.bitcast(a, flt, 'k')
    self.assertFalse(block.is_terminated)
    self.check_block(block, '            my_block:\n                %"c" = trunc i32 %".1" to i8\n                %"d" = zext i8 %"c" to i32\n                %"e" = sext i8 %"c" to i32\n                %"fb" = fptrunc double %".3" to float\n                %"fc" = fpext float %"fb" to double\n                %"g" = fptoui double %".3" to i32\n                %"h" = fptosi double %".3" to i8\n                %"fd" = uitofp i32 %"g" to float\n                %"fe" = sitofp i8 %"h" to double\n                %"i" = ptrtoint i32* %".4" to i32\n                %"j" = inttoptr i32 %"i" to i8*\n                %"k" = bitcast i32 %".1" to float\n            ')