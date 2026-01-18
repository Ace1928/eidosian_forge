from inspect import Parameter, Signature, getsource
from typing import Any, Dict, List, cast
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import inspect, logging
def not_suppressed(argtypes: List[ast.AST]=[]) -> bool:
    """Check given *argtypes* is suppressed type_comment or not."""
    if len(argtypes) == 0:
        return False
    elif len(argtypes) == 1 and ast_unparse(argtypes[0]) == '...':
        return False
    else:
        return True