from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def get_code_complexity(code, threshold=7, filename='stdin'):
    try:
        tree = compile(code, filename, 'exec', ast.PyCF_ONLY_AST)
    except SyntaxError:
        e = sys.exc_info()[1]
        sys.stderr.write('Unable to parse %s: %s\n' % (filename, e))
        return 0
    complx = []
    McCabeChecker.max_complexity = threshold
    for lineno, offset, text, check in McCabeChecker(tree, filename).run():
        complx.append('%s:%d:1: %s' % (filename, lineno, text))
    if len(complx) == 0:
        return 0
    print('\n'.join(complx))
    return len(complx)