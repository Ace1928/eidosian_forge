import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_metadata_string(self):
    mod = self.module()
    mod.add_metadata(['"\\$'])
    self.assert_ir_line('!0 = !{ !"\\22\\5c$" }', mod)