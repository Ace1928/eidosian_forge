import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_fence(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    with self.assertRaises(ValueError) as raises:
        builder.fence('monotonic', None)
    self.assertIn('Invalid fence ordering "monotonic"!', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        builder.fence(None, 'monotonic')
    self.assertIn('Invalid fence ordering "None"!', str(raises.exception))
    builder.fence('acquire', None)
    builder.fence('release', 'singlethread')
    builder.fence('acq_rel', 'singlethread')
    builder.fence('seq_cst')
    builder.ret_void()
    self.check_block(block, '            my_block:\n                fence acquire\n                fence syncscope("singlethread") release\n                fence syncscope("singlethread") acq_rel\n                fence seq_cst\n                ret void\n            ')