import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
@staticmethod
def create_ast(filename):
    """Create an Abstract Syntax Tree on which a visitor can be run"""
    node = None
    with tokenize.open(filename) as file:
        node = ast.parse(file.read(), mode='exec')
    return node