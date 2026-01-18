import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def assert_ir_line(self, line, mod):
    lines = [line.strip() for line in str(mod).splitlines()]
    self.assertIn(line, lines)