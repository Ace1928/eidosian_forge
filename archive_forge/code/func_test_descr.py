import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_descr(self):
    block = self.block(name='my_block')
    self.assertEqual(self.descr(block), 'my_block:\n')
    block.instructions.extend(['a', 'b'])
    self.assertEqual(self.descr(block), 'my_block:\n  a\n  b\n')