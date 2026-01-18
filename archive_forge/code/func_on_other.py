import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
def on_other(self, states, stmt):
    newdef = self._fix_var(states, stmt, self._cache_list_vars.get(stmt))
    if newdef is not None and newdef.target is not ir.UNDEFINED:
        if states['varname'] != newdef.target.name:
            replmap = {states['varname']: newdef.target}
            stmt = copy(stmt)
            ir_utils.replace_vars_stmt(stmt, replmap)
    return stmt