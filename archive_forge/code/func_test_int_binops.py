import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_int_binops(self):
    one = ir.Constant(int32, 1)
    two = ir.Constant(int32, 2)
    oracle = {one.shl: 'shl', one.lshr: 'lshr', one.ashr: 'ashr', one.add: 'add', one.sub: 'sub', one.mul: 'mul', one.udiv: 'udiv', one.sdiv: 'sdiv', one.urem: 'urem', one.srem: 'srem', one.or_: 'or', one.and_: 'and', one.xor: 'xor'}
    for fn, irop in oracle.items():
        self.assertEqual(str(fn(two)), irop + ' (i32 1, i32 2)')
    oracle = {'==': 'eq', '!=': 'ne', '>': 'ugt', '>=': 'uge', '<': 'ult', '<=': 'ule'}
    for cop, cond in oracle.items():
        actual = str(one.icmp_unsigned(cop, two))
        expected = 'icmp ' + cond + ' (i32 1, i32 2)'
        self.assertEqual(actual, expected)
    oracle = {'==': 'eq', '!=': 'ne', '>': 'sgt', '>=': 'sge', '<': 'slt', '<=': 'sle'}
    for cop, cond in oracle.items():
        actual = str(one.icmp_signed(cop, two))
        expected = 'icmp ' + cond + ' (i32 1, i32 2)'
        self.assertEqual(actual, expected)