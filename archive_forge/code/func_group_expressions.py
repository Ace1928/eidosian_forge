import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def group_expressions(expressions: Iterable[Tuple[ast.expr, Any]]) -> List[Tuple[List[ast.expr], Any]]:
    """
    Organise expression nodes and their values such that equivalent nodes are together.
    Two nodes are considered equivalent if they have the same structure,
    ignoring context (Load, Store, or Delete) and location (lineno, col_offset).
    For example, this will group together the same variable name mentioned multiple times in an expression.

    This will not check the values of the nodes. Equivalent nodes should have the same values,
    unless threads are involved.

    This is a low level API, typically you will use `interesting_expressions_grouped`.

    :param expressions: pairs of AST expressions and their values, as obtained from
                          `Evaluator.find_expressions`, or `(node, evaluator[node])`.
    :return: A list of pairs (tuples) containing:
                - A list of equivalent AST expressions
                - The value of the first expression node
                   (which should be the same for all nodes, unless threads are involved)
    """
    result = {}
    for node, value in expressions:
        dump = ast.dump(copy_ast_without_context(node))
        result.setdefault(dump, ([], value))[0].append(node)
    return list(result.values())