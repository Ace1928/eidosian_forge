import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def guard_iter(self, node):
    """
		Converts:
			for x in expr
		to
			for x in __rl_getiter__(expr)

		Also used for
		* list comprehensions
		* dict comprehensions
		* set comprehensions
		* generator expresions
		"""
    node = self.visit_children(node)
    if isinstance(node.target, ast.Tuple):
        spec = self.gen_unpack_spec(node.target)
        new_iter = ast.Call(func=ast.Name('__rl_iter_unpack_sequence__', ast.Load()), args=[node.iter, spec, ast.Name('__rl_getiter__', ast.Load())], keywords=[])
    else:
        new_iter = ast.Call(func=ast.Name('__rl_getiter__', ast.Load()), args=[node.iter], keywords=[])
    copy_locations(new_iter, node.iter)
    node.iter = new_iter
    return node