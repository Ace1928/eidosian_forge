import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_extract_insert_value(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    tp_inner = ir.LiteralStructType([int32, int1])
    tp_outer = ir.LiteralStructType([int8, tp_inner])
    c_inner = ir.Constant(tp_inner, (ir.Constant(int32, 4), ir.Constant(int1, True)))
    c = builder.extract_value(c_inner, 0, name='c')
    d = builder.insert_value(c_inner, a, 0, name='d')
    e = builder.insert_value(d, ir.Constant(int1, False), 1, name='e')
    self.assertEqual(d.type, tp_inner)
    self.assertEqual(e.type, tp_inner)
    p_outer = builder.alloca(tp_outer, name='ptr')
    j = builder.load(p_outer, name='j')
    k = builder.extract_value(j, 0, name='k')
    l = builder.extract_value(j, 1, name='l')
    m = builder.extract_value(j, (1, 0), name='m')
    n = builder.extract_value(j, (1, 1), name='n')
    o = builder.insert_value(j, l, 1, name='o')
    p = builder.insert_value(j, a, (1, 0), name='p')
    self.assertEqual(k.type, int8)
    self.assertEqual(l.type, tp_inner)
    self.assertEqual(m.type, int32)
    self.assertEqual(n.type, int1)
    self.assertEqual(o.type, tp_outer)
    self.assertEqual(p.type, tp_outer)
    with self.assertRaises(TypeError):
        builder.extract_value(p_outer, 0)
    with self.assertRaises(TypeError):
        builder.extract_value(c_inner, (0, 0))
    with self.assertRaises(TypeError):
        builder.extract_value(c_inner, 5)
    with self.assertRaises(TypeError):
        builder.insert_value(a, b, 0)
    with self.assertRaises(TypeError):
        builder.insert_value(c_inner, a, 1)
    self.check_block(block, '            my_block:\n                %"c" = extractvalue {i32, i1} {i32 4, i1 true}, 0\n                %"d" = insertvalue {i32, i1} {i32 4, i1 true}, i32 %".1", 0\n                %"e" = insertvalue {i32, i1} %"d", i1 false, 1\n                %"ptr" = alloca {i8, {i32, i1}}\n                %"j" = load {i8, {i32, i1}}, {i8, {i32, i1}}* %"ptr"\n                %"k" = extractvalue {i8, {i32, i1}} %"j", 0\n                %"l" = extractvalue {i8, {i32, i1}} %"j", 1\n                %"m" = extractvalue {i8, {i32, i1}} %"j", 1, 0\n                %"n" = extractvalue {i8, {i32, i1}} %"j", 1, 1\n                %"o" = insertvalue {i8, {i32, i1}} %"j", {i32, i1} %"l", 1\n                %"p" = insertvalue {i8, {i32, i1}} %"j", i32 %".1", 1, 0\n            ')