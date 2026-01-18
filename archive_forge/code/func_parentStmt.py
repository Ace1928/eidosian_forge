from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def parentStmt(self, node):
    return self.parentInstance(node, ast.stmt)