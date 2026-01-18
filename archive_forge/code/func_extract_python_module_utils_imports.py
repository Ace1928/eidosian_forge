from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def extract_python_module_utils_imports(path: str, module_utils: set[str]) -> set[str]:
    """Return a list of module_utils imports found in the specified source file."""
    code = read_binary_file(path)
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        display.warning('%s:%s Syntax error extracting module_utils imports: %s' % (path, ex.lineno, ex.msg))
        return set()
    finder = ModuleUtilFinder(path, module_utils)
    finder.visit(tree)
    return finder.imports