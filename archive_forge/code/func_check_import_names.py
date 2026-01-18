import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def check_import_names(self, node):
    """Check the names being imported.

		This is a protection against rebinding dunder names like
		__rl_getitem__,__rl_set__ via imports.

		=> 'from _a import x' is ok, because '_a' is not added to the scope.
		"""
    for name in node.names:
        if '*' in name.name:
            self.error(node, '"*" imports are not allowed.')
        self.isAllowedName(node, name.name)
        if name.asname:
            self.isAllowedName(node, name.asname)
    return self.visit_children(node)