import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def collect_try_except_info_from_contents(contents, filename='<unknown>'):
    ast = ast_module.parse(contents, filename)
    visitor = _Visitor()
    visitor.visit(ast)
    return visitor.try_except_infos