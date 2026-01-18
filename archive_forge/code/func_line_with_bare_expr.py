from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def line_with_bare_expr(code):
    """return None or else 0-based line number of code on which
    a bare expression appeared.
    """
    tree = ast.parse(code)
    try:
        BareExpr.visit(tree)
    except AssertionError as msg:
        assert msg.args
        msg = msg.args[0]
        assert msg.startswith(message_bare_expr.split(':', 1)[0])
        return int(msg.rsplit(' ', 1)[1].rstrip('.'))