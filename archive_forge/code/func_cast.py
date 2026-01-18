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
@classmethod
def cast(cls, value, loc):
    """
        A node for implicit casting at the return statement
        """
    assert isinstance(value, Var)
    assert isinstance(loc, Loc)
    op = 'cast'
    return cls(op=op, value=value, loc=loc)