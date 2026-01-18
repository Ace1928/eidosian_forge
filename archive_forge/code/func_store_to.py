import logging
import sys
from weakref import ref as weakref_ref
import gc
import math
from pyomo.common import timing
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pympler, pympler_available
from pyomo.common.deprecation import deprecated
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.set import Set
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.label import CNameLabeler, CuidLabeler
from pyomo.dataportal.DataPortal import DataPortal
from pyomo.opt.results import Solution, SolverStatus, UndefinedData
from contextlib import nullcontext
from io import StringIO
def store_to(self, results, cuid=False, skip_stale_vars=False):
    """
        Return a Solution() object that is populated with the values in the model.
        """
    instance = self._instance()
    results.solution.clear()
    results._smap_id = None
    for soln_ in self.solutions:
        soln = Solution()
        soln._cuid = cuid
        for key, val in soln_._metadata.items():
            setattr(soln, key, val)
        if cuid:
            labeler = CuidLabeler()
        else:
            labeler = CNameLabeler()
        sm = SymbolMap()
        entry = soln_._entry['objective']
        for obj in instance.component_data_objects(Objective, active=True):
            vals = entry.get(id(obj), None)
            if vals is None:
                vals = {}
            else:
                vals = vals[1]
            vals['Value'] = value(obj)
            soln.objective[sm.getSymbol(obj, labeler)] = vals
        entry = soln_._entry['variable']
        for obj in instance.component_data_objects(Var, active=True):
            if obj.stale and skip_stale_vars:
                continue
            vals = entry.get(id(obj), None)
            if vals is None:
                vals = {}
            else:
                vals = vals[1]
            vals['Value'] = value(obj)
            soln.variable[sm.getSymbol(obj, labeler)] = vals
        entry = soln_._entry['constraint']
        for obj in instance.component_data_objects(Constraint, active=True):
            vals = entry.get(id(obj), None)
            if vals is None:
                continue
            else:
                vals = vals[1]
            soln.constraint[sm.getSymbol(obj, labeler)] = vals
        results.solution.insert(soln)