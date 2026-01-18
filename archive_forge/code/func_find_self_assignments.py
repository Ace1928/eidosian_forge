from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def find_self_assignments(s):
    """Returns a list of "bad" assignments: if there are instances
    of assigning to the first argument of the class method (except
    for staticmethod's).
    """
    t = [n for n in ast.parse(s).body if isinstance(n, ast.ClassDef)]
    bad = []
    for c in t:
        for n in c.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            if any((d.id == 'staticmethod' for d in n.decorator_list if isinstance(d, ast.Name))):
                continue
            if n.name == '__new__':
                continue
            if not n.args.args:
                continue
            first_arg = n.args.args[0].arg
            for m in ast.walk(n):
                if isinstance(m, ast.Assign):
                    for a in m.targets:
                        if isinstance(a, ast.Name) and a.id == first_arg:
                            bad.append(m)
                        elif isinstance(a, ast.Tuple) and any((q.id == first_arg for q in a.elts if isinstance(q, ast.Name))):
                            bad.append(m)
    return bad