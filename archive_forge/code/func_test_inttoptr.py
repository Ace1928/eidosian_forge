import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_inttoptr(self):
    one = ir.Constant(int32, 1)
    pi = ir.Constant(flt, 3.14)
    c = one.inttoptr(int64.as_pointer())
    self.assertRaises(TypeError, one.inttoptr, int64)
    self.assertRaises(TypeError, pi.inttoptr, int64.as_pointer())
    self.assertEqual(str(c), 'inttoptr (i32 1 to i64*)')