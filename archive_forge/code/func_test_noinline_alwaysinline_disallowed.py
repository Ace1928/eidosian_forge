import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_noinline_alwaysinline_disallowed(self):
    module = self.module()
    func = self.function(module)
    func.attributes.add('noinline')
    msg = "Can't have alwaysinline and noinline"
    with self.assertRaisesRegex(ValueError, msg):
        func.attributes.add('alwaysinline')