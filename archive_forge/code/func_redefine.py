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
def redefine(self, name, loc, rename=True):
    """
        Redefine if the name is already defined
        """
    if name not in self.localvars:
        return self.define(name, loc)
    elif not rename:
        return self.localvars.get(name)
    else:
        while True:
            ct = self.redefined[name]
            self.redefined[name] = ct + 1
            newname = '%s.%d' % (name, ct + 1)
            try:
                res = self.define(newname, loc)
            except RedefinedError:
                continue
            else:
                self.var_redefinitions[name].add(newname)
            return res