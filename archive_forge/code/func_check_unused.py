from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def check_unused(self, node, skipped_types=()):
    for local_def in self.defuses.locals[node]:
        if not local_def.users():
            if local_def.name() == '_':
                continue
            if isinstance(local_def.node, skipped_types):
                continue
            location = local_def.node
            while not hasattr(location, 'lineno'):
                location = self.ancestors.parent(location)
            if isinstance(location, ast.ImportFrom):
                if location.module == '__future__':
                    continue
            print("W: '{}' is defined but not used at {}:{}:{}".format(local_def.name(), self.filename, location.lineno, location.col_offset))