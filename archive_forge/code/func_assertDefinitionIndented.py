import difflib
import inspect
import re
import unittest
from code import compile_command as compiler
from functools import partial
from bpython.curtsiesfrontend.interpreter import code_finished_will_parse
from bpython.curtsiesfrontend.preprocess import preprocess
from bpython.test.fodder import original, processed
def assertDefinitionIndented(self, obj):
    name = obj.__name__
    obj2 = getattr(processed, name)
    orig = inspect.getsource(obj)
    xformed = inspect.getsource(obj2)
    self.assertShowWhitespaceEqual(preproc(orig), xformed)
    self.assertCompiles(xformed)