import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_fneg(self):
    one = ir.Constant(flt, 1)
    self.assertEqual(str(one.fneg()), 'fneg (float 0x3ff0000000000000)')