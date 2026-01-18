import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
def nice_explanation(self):
    return _format_explanation(self.explanation)