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
def find_variable_assignment(self, name):
    """
        Returns the assignment inst associated with variable "name", None if
        it cannot be found.
        """
    for x in self.find_insts(cls=Assign):
        if x.target.name == name:
            return x
    return None