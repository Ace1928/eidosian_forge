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
def add_solution(self, solution, smap_id, delete_symbol_map=True, cache=None, ignore_invalid_labels=False, ignore_missing_symbols=True, default_variable_value=None):
    instance = self._instance()
    soln = ModelSolution()
    soln._metadata['status'] = solution.status
    if not type(solution.message) is UndefinedData:
        soln._metadata['message'] = solution.message
    if not type(solution.gap) is UndefinedData:
        soln._metadata['gap'] = solution.gap
    if smap_id is None:
        if cache is None:
            cache = {}
        if solution._cuid:
            if len(cache) == 0:
                for obj in instance.component_data_objects(Var):
                    cache[ComponentUID(obj)] = obj
                for obj in instance.component_data_objects(Objective, active=True):
                    cache[ComponentUID(obj)] = obj
                for obj in instance.component_data_objects(Constraint, active=True):
                    cache[ComponentUID(obj)] = obj
            for name in ['problem', 'objective', 'variable', 'constraint']:
                tmp = soln._entry[name]
                for cuid, val in getattr(solution, name).items():
                    obj = cache.get(cuid, None)
                    if obj is None:
                        if ignore_invalid_labels:
                            continue
                        raise RuntimeError('CUID %s is missing from model %s' % (str(cuid), instance.name))
                    tmp[id(obj)] = (obj, val)
        else:
            if len(cache) == 0:
                for obj in instance.component_data_objects(Var):
                    cache[obj.name] = obj
                for obj in instance.component_data_objects(Objective, active=True):
                    cache[obj.name] = obj
                for obj in instance.component_data_objects(Constraint, active=True):
                    cache[obj.name] = obj
            for name in ['problem', 'objective', 'variable', 'constraint']:
                tmp = soln._entry[name]
                for symb, val in getattr(solution, name).items():
                    obj = cache.get(symb, None)
                    if obj is None:
                        if ignore_invalid_labels:
                            continue
                        raise RuntimeError('Symbol %s is missing from model %s' % (symb, instance.name))
                    tmp[id(obj)] = (obj, val)
    else:
        smap = self.symbol_map[smap_id]
        for name in ['problem', 'objective', 'variable', 'constraint']:
            tmp = soln._entry[name]
            for symb, val in getattr(solution, name).items():
                if symb in smap.bySymbol:
                    obj = smap.bySymbol[symb]
                elif symb in smap.aliases:
                    obj = smap.aliases[symb]
                elif ignore_missing_symbols:
                    continue
                else:
                    raise RuntimeError('ERROR: Symbol %s is missing from model %s when loading with a symbol map!' % (symb, instance.name))
                tmp[id(obj)] = (obj, val)
        if delete_symbol_map:
            self.delete_symbol_map(smap_id)
    tmp = soln._entry['variable']
    for vdata in instance.component_data_objects(Var):
        id_ = id(vdata)
        if vdata.fixed:
            tmp[id_] = (vdata, {'Value': vdata.value})
        elif default_variable_value is not None and smap_id is not None and (id_ in smap.byObject) and (id_ not in tmp):
            tmp[id_] = (vdata, {'Value': default_variable_value})
    self.solutions.append(soln)
    return len(self.solutions) - 1