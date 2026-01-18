import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def check_module_body(self, module, asm):
    expected = self._normalize_asm(asm)
    actual = module._stringify_body()
    self.assertEqual(actual.strip(), expected.strip())