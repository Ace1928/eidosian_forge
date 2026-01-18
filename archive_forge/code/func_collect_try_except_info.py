import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def collect_try_except_info(co, use_func_first_line=False):
    """
        Note: if the filename is available and we can get the source,
        `collect_try_except_info_from_source` is preferred (this is kept as
        a fallback for cases where sources aren't available).
        """
    return []