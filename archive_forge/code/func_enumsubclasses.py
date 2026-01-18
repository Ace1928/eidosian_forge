import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
def enumsubclasses(cls):
    for subcls in cls.__subclasses__():
        for subsubclass in enumsubclasses(subcls):
            yield subsubclass
    yield cls