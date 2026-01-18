import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_flt_binops(self):
    one = ir.Constant(flt, 1)
    two = ir.Constant(flt, 2)
    oracle = {one.fadd: 'fadd', one.fsub: 'fsub', one.fmul: 'fmul', one.fdiv: 'fdiv', one.frem: 'frem'}
    for fn, irop in oracle.items():
        actual = str(fn(two))
        expected = irop + ' (float 0x3ff0000000000000, float 0x4000000000000000)'
        self.assertEqual(actual, expected)
    oracle = {'==': 'oeq', '!=': 'one', '>': 'ogt', '>=': 'oge', '<': 'olt', '<=': 'ole'}
    for cop, cond in oracle.items():
        actual = str(one.fcmp_ordered(cop, two))
        expected = 'fcmp ' + cond + ' (float 0x3ff0000000000000, float 0x4000000000000000)'
        self.assertEqual(actual, expected)
    oracle = {'==': 'ueq', '!=': 'une', '>': 'ugt', '>=': 'uge', '<': 'ult', '<=': 'ule'}
    for cop, cond in oracle.items():
        actual = str(one.fcmp_unordered(cop, two))
        expected = 'fcmp ' + cond + ' (float 0x3ff0000000000000, float 0x4000000000000000)'
        self.assertEqual(actual, expected)