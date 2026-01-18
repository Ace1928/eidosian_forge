import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def literal_eval(node_or_string):
    """
    Evaluate an expression node or a string containing only a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, and None.

    Caution: A complex expression can overflow the C stack and cause a crash.
    """
    if isinstance(node_or_string, str):
        node_or_string = parse(node_or_string.lstrip(' \t'), mode='eval')
    if isinstance(node_or_string, Expression):
        node_or_string = node_or_string.body

    def _raise_malformed_node(node):
        msg = 'malformed node or string'
        if (lno := getattr(node, 'lineno', None)):
            msg += f' on line {lno}'
        raise ValueError(msg + f': {node!r}')

    def _convert_num(node):
        if not isinstance(node, Constant) or type(node.value) not in (int, float, complex):
            _raise_malformed_node(node)
        return node.value

    def _convert_signed_num(node):
        if isinstance(node, UnaryOp) and isinstance(node.op, (UAdd, USub)):
            operand = _convert_num(node.operand)
            if isinstance(node.op, UAdd):
                return +operand
            else:
                return -operand
        return _convert_num(node)

    def _convert(node):
        if isinstance(node, Constant):
            return node.value
        elif isinstance(node, Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, List):
            return list(map(_convert, node.elts))
        elif isinstance(node, Set):
            return set(map(_convert, node.elts))
        elif isinstance(node, Call) and isinstance(node.func, Name) and (node.func.id == 'set') and (node.args == node.keywords == []):
            return set()
        elif isinstance(node, Dict):
            if len(node.keys) != len(node.values):
                _raise_malformed_node(node)
            return dict(zip(map(_convert, node.keys), map(_convert, node.values)))
        elif isinstance(node, BinOp) and isinstance(node.op, (Add, Sub)):
            left = _convert_signed_num(node.left)
            right = _convert_num(node.right)
            if isinstance(left, (int, float)) and isinstance(right, complex):
                if isinstance(node.op, Add):
                    return left + right
                else:
                    return left - right
        return _convert_signed_num(node)
    return _convert(node_or_string)