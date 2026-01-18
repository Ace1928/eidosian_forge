from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
def find_exprs(self, op=None):
    """
        Iterate over exprs of the given *op* in this block.
        """
    for inst in self.body:
        if isinstance(inst, Assign):
            expr = inst.value
            if isinstance(expr, Expr):
                if op is None or expr.op == op:
                    yield expr