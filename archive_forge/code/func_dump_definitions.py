from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def dump_definitions(self, node, ignore_builtins=True):
    if isinstance(node, ast.Module) and (not ignore_builtins):
        builtins = {d for d in self._builtins.values()}
        return sorted((d.name() for d in self.locals[node] if d not in builtins))
    else:
        return sorted((d.name() for d in self.locals[node]))