import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def find_expressions(self, root: ast.AST) -> Iterable[Tuple[ast.expr, Any]]:
    """
        Find all expressions in the given tree that can be safely evaluated.
        This is a low level API, typically you will use `interesting_expressions_grouped`.

        :param root: any AST node
        :return: generator of pairs (tuples) of expression nodes and their corresponding values.
        """
    for node in ast.walk(root):
        if not isinstance(node, ast.expr):
            continue
        try:
            value = self[node]
        except CannotEval:
            continue
        yield (node, value)