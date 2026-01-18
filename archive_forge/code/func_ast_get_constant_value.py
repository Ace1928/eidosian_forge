import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
def ast_get_constant_value(node: ast.AST) -> T.Any:
    """Return the constant's value if the given node is a constant."""
    return getattr(node, ast_constant_attr[node.__class__])