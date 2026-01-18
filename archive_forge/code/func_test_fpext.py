import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_fpext(self):
    c = ir.Constant(flt, 1).fpext(dbl)
    self.assertEqual(str(c), 'fpext (float 0x3ff0000000000000 to double)')