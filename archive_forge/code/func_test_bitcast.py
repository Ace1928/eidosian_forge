import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_bitcast(self):
    m = self.module()
    gv = ir.GlobalVariable(m, int32, 'myconstant')
    c = gv.bitcast(int64.as_pointer())
    self.assertEqual(str(c), 'bitcast (i32* @"myconstant" to i64*)')