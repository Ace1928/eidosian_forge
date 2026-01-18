import difflib
import inspect
import re
import unittest
from code import compile_command as compiler
from functools import partial
from bpython.curtsiesfrontend.interpreter import code_finished_will_parse
from bpython.curtsiesfrontend.preprocess import preprocess
from bpython.test.fodder import original, processed
def test_indent_empty_lines_nops(self):
    self.assertEqual(preproc('hello'), 'hello')
    self.assertEqual(preproc('hello\ngoodbye'), 'hello\ngoodbye')
    self.assertEqual(preproc('a\n    b\nc\n'), 'a\n    b\nc\n')