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
def get_assignee(self, rhs_value, in_blocks=None):
    """
        Finds the assignee for a given RHS value. If in_blocks is given the
        search will be limited to the specified blocks.
        """
    if in_blocks is None:
        blocks = self.blocks.values()
    elif isinstance(in_blocks, int):
        blocks = [self.blocks[in_blocks]]
    else:
        blocks = [self.blocks[blk] for blk in list(in_blocks)]
    assert isinstance(rhs_value, AbstractRHS)
    for blk in blocks:
        for assign in blk.find_insts(Assign):
            if assign.value == rhs_value:
                return assign.target
    raise ValueError('Could not find an assignee for %s' % rhs_value)