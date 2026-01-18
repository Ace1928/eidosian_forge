import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def gen_unpack_wrapper(self, node, target, ctx='store'):
    """Helper function to protect tuple unpacks.

		node: used to copy the locations for the new nodes.
		target: is the tuple which must be protected.
		ctx: Defines the context of the returned temporary node.

		It returns a tuple with two element.

		Element 1: Is a temporary name node which must be used to
				   replace the target.
				   The context (store, param) is defined
				   by the 'ctx' parameter..

		Element 2: Is a try .. finally where the body performs the
				   protected tuple unpack of the temporary variable
				   into the original target.
		"""
    tnam = self.tmpName
    converter = self.protect_unpack_sequence(target, ast.Name(tnam, ast.Load()))
    try_body = [ast.Assign(targets=[target], value=converter)]
    finalbody = [self.gen_del_stmt(tnam)]
    cleanup = ast.Try(body=try_body, finalbody=finalbody, handlers=[], orelse=[])
    if ctx == 'store':
        ctx = ast.Store()
    elif ctx == 'param':
        ctx = ast.Param()
    else:
        raise NotImplementedError('bad ctx "%s"' % type(ctx))
    tmp_target = ast.Name(tnam, ctx)
    copy_locations(tmp_target, node)
    copy_locations(cleanup, node)
    return (tmp_target, cleanup)