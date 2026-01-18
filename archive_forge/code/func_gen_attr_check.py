import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def gen_attr_check(self, node, attr_name):
    """Check if 'attr_name' is allowed on the object in node.

		It generates (_getattr_(node, attr_name) and node).
		"""
    call_getattr = ast.Call(func=ast.Name('__rl_getattr__', ast.Load()), args=[node, ast.Str(attr_name)], keywords=[])
    return ast.BoolOp(op=ast.And(), values=[call_getattr, node])