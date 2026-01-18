import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def rl_extended_literal_eval(expr, safe_callables=None, safe_names=None):
    if safe_callables is None:
        safe_callables = {}
    if safe_names is None:
        safe_names = {}
    safe_names = safe_names.copy()
    safe_names.update({'None': None, 'True': True, 'False': False})
    safe_names = types.MappingProxyType(safe_names)
    safe_callables = types.MappingProxyType(safe_callables)
    if isinstance(expr, str):
        expr = ast.parse(expr, mode='eval')
    if isinstance(expr, ast.Expression):
        expr = expr.body
    try:
        ast.NameConstant
        safe_test = lambda n: isinstance(n, ast.NameConstant) or (isinstance(n, ast.Name) and n.id in safe_names)
        safe_extract = lambda n: n.value if isinstance(n, ast.NameConstant) else safe_names[n.id]
    except AttributeError:
        safe_test = lambda n: isinstance(n, ast.Name) and n.id in safe_names
        safe_extract = lambda n: safe_names[n.id]

    def _convert(node):
        if isinstance(node, (ast.Str, ast.Bytes)):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, ast.List):
            return list(map(_convert, node.elts))
        elif isinstance(node, ast.Dict):
            return dict(((_convert(k), _convert(v)) for k, v in zip(node.keys, node.values)))
        elif safe_test(node):
            return safe_extract(node)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, (ast.Num, ast.UnaryOp, ast.BinOp)):
            operand = _convert(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            else:
                return -operand
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)) and isinstance(node.right, (ast.Num, ast.UnaryOp, ast.BinOp)) and isinstance(node.right.n, complex) and isinstance(node.left, (ast.Num, ast.UnaryOp, astBinOp)):
            left = _convert(node.left)
            right = _convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            else:
                return left - right
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id in safe_callables):
            return safe_callables[node.func.id](*[_convert(n) for n in node.args], **{kw.arg: _convert(kw.value) for kw in node.keywords})
        raise ValueError('Bad expression')
    return _convert(expr)