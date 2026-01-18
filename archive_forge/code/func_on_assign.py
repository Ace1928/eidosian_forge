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
def on_assign(self, states, assign):
    rhs = assign.value
    if isinstance(rhs, ir.Inst):
        newdef = self._fix_var(states, assign, self._cache_list_vars.get(assign.value))
        if newdef is not None and newdef.target is not ir.UNDEFINED:
            if states['varname'] != newdef.target.name:
                replmap = {states['varname']: newdef.target}
                rhs = copy(rhs)
                ir_utils.replace_vars_inner(rhs, replmap)
                return ir.Assign(target=assign.target, value=rhs, loc=assign.loc)
    elif isinstance(rhs, ir.Var):
        newdef = self._fix_var(states, assign, [rhs])
        if newdef is not None and newdef.target is not ir.UNDEFINED:
            if states['varname'] != newdef.target.name:
                return ir.Assign(target=assign.target, value=newdef.target, loc=assign.loc)
    return assign