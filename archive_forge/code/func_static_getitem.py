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
def static_getitem(cls, value, index, index_var, loc):
    assert isinstance(value, Var)
    assert index_var is None or isinstance(index_var, Var)
    assert isinstance(loc, Loc)
    op = 'static_getitem'
    fn = operator.getitem
    return cls(op=op, loc=loc, value=value, index=index, index_var=index_var, fn=fn)