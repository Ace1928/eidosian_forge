import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def check_constant(tp, i, expected):
    actual = tp.gep(ir.Constant(int32, i))
    self.assertEqual(actual, expected)