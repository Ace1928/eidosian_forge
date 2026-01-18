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
def get_or_define(self, name, loc):
    if name in self.redefined:
        name = '%s.%d' % (name, self.redefined[name])
    if name not in self.localvars:
        return self.define(name, loc)
    else:
        return self.localvars.get(name)