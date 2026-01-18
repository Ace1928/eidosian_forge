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
def get_pad(ablock, l):
    pointer = '-> '
    sp = len(pointer) * ' '
    pad = []
    nstmt = len(ablock)
    for i in range(nstmt):
        if i in tmp:
            item = pointer
        elif i >= l:
            item = pointer
        else:
            item = sp
        pad.append(item)
    return pad