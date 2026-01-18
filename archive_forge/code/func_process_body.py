from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def process_body(self, stmts):
    deadcode = False
    for stmt in stmts:
        if isinstance(stmt, (ast.Break, ast.Continue, ast.Raise)):
            if not deadcode:
                deadcode = True
                self.deadcode += 1
        self.visit(stmt)
    if deadcode:
        self.deadcode -= 1