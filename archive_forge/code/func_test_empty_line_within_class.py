import difflib
import inspect
import re
import unittest
from code import compile_command as compiler
from functools import partial
from bpython.curtsiesfrontend.interpreter import code_finished_will_parse
from bpython.curtsiesfrontend.preprocess import preprocess
from bpython.test.fodder import original, processed
def test_empty_line_within_class(self):
    self.assertIndented(original.BlankLineInFunction)