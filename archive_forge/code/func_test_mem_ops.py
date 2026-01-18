import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_mem_ops(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b, z = builder.function.args[:3]
    c = builder.alloca(int32, name='c')
    d = builder.alloca(int32, size=42, name='d')
    e = builder.alloca(dbl, size=a, name='e')
    e.align = 8
    self.assertEqual(e.type, ir.PointerType(dbl))
    ee = builder.store(z, e)
    self.assertEqual(ee.type, ir.VoidType())
    f = builder.store(b, c)
    self.assertEqual(f.type, ir.VoidType())
    g = builder.load(c, 'g')
    self.assertEqual(g.type, int32)
    h = builder.store(b, c, align=1)
    self.assertEqual(h.type, ir.VoidType())
    i = builder.load(c, 'i', align=1)
    self.assertEqual(i.type, int32)
    j = builder.store_atomic(b, c, ordering='seq_cst', align=4)
    self.assertEqual(j.type, ir.VoidType())
    k = builder.load_atomic(c, ordering='seq_cst', align=4, name='k')
    self.assertEqual(k.type, int32)
    with self.assertRaises(TypeError):
        builder.store(b, a)
    with self.assertRaises(TypeError):
        builder.load(b)
    with self.assertRaises(TypeError) as cm:
        builder.store(b, e)
    self.assertEqual(str(cm.exception), 'cannot store i32 to double*: mismatching types')
    self.check_block(block, '            my_block:\n                %"c" = alloca i32\n                %"d" = alloca i32, i32 42\n                %"e" = alloca double, i32 %".1", align 8\n                store double %".3", double* %"e"\n                store i32 %".2", i32* %"c"\n                %"g" = load i32, i32* %"c"\n                store i32 %".2", i32* %"c", align 1\n                %"i" = load i32, i32* %"c", align 1\n                store atomic i32 %".2", i32* %"c" seq_cst, align 4\n                %"k" = load atomic i32, i32* %"c" seq_cst, align 4\n            ')