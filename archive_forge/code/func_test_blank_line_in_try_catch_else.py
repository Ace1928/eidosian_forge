import difflib
import inspect
import re
import unittest
from code import compile_command as compiler
from functools import partial
from bpython.curtsiesfrontend.interpreter import code_finished_will_parse
from bpython.curtsiesfrontend.preprocess import preprocess
from bpython.test.fodder import original, processed
@unittest.skip('More advanced technique required: need to try compiling and backtracking')
def test_blank_line_in_try_catch_else(self):
    self.assertIndented('blank_line_in_try_catch_else')