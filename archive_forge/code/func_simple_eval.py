import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def simple_eval(node_or_string, namespace=None):
    """
    Safely evaluate an expression node or a string containing a Python
    expression without triggering any user code.

    The string or node provided may only consist of:
    * the following Python literal structures: strings, numbers, tuples,
        lists, dicts, and sets
    * variable names causing lookups in the passed in namespace or builtins
    * getitem calls using the [] syntax on objects of the types above

    Like Python 3's literal_eval, unary and binary + and - operations are
    allowed on all builtin numeric types.

    The optional namespace dict-like ought not to cause side effects on lookup.
    """
    if namespace is None:
        namespace = {}
    if isinstance(node_or_string, str):
        node_or_string = ast.parse(node_or_string, mode='eval')
    if isinstance(node_or_string, ast.Expression):
        node_or_string = node_or_string.body

    def _convert(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif not _is_py38 and isinstance(node, _string_type_nodes):
            return node.s
        elif not _is_py38 and isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, ast.List):
            return list(map(_convert, node.elts))
        elif isinstance(node, ast.Dict):
            return {_convert(k): _convert(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Set):
            return set(map(_convert, node.elts))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'set') and (node.args == node.keywords == []):
            return set()
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'tuple') and (node.args == node.keywords == []):
            return tuple()
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'list') and (node.args == node.keywords == []):
            return list()
        elif isinstance(node, _name_type_nodes):
            try:
                return namespace[node.id]
            except KeyError:
                try:
                    return getattr(builtins, node.id)
                except AttributeError:
                    raise EvaluationError("can't lookup %s" % node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _convert(node.operand)
            if not type(operand) in _numeric_types:
                raise ValueError('unary + and - only allowed on builtin nums')
            if isinstance(node.op, ast.UAdd):
                return +operand
            else:
                return -operand
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = _convert(node.left)
            right = _convert(node.right)
            if not (isinstance(left, _numeric_types) and isinstance(right, _numeric_types)):
                raise ValueError('binary + and - only allowed on builtin nums')
            if isinstance(node.op, ast.Add):
                return left + right
            else:
                return left - right
        elif not _is_py39 and isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Index):
            obj = _convert(node.value)
            index = _convert(node.slice.value)
            return safe_getitem(obj, index)
        elif _is_py39 and isinstance(node, ast.Subscript) and isinstance(node.slice, (ast.Constant, ast.Name)):
            obj = _convert(node.value)
            index = _convert(node.slice)
            return safe_getitem(obj, index)
        if isinstance(node, ast.Attribute):
            obj = _convert(node.value)
            attr = node.attr
            return getattr_safe(obj, attr)
        raise ValueError(f'malformed node or string: {node!r}')
    return _convert(node_or_string)