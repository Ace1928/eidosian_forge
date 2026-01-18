from os.path import join, dirname, abspath
import json
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core import Suffix, Var, Constraint, Objective
from pyomo.opt import ProblemFormat, SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def save_current_solution(self, filename, **kwds):
    """Save the solution in a specified file name"""
    assert self.model is not None
    model = self.model
    suffixes = dict(((suffix, getattr(model, suffix)) for suffix in kwds.pop('suffixes', [])))
    for suf in suffixes.values():
        assert suf.import_enabled()
    with open(filename, 'w') as f:
        soln = {}
        for block in model.block_data_objects():
            soln[block.name] = {}
            for suffix_name, suffix in suffixes.items():
                if suffix.get(block) is not None:
                    soln[block.name][suffix_name] = suffix.get(block)
        for var in model.component_data_objects(Var):
            soln[var.name] = {}
            soln[var.name]['value'] = var.value
            soln[var.name]['stale'] = var.stale
            for suffix_name, suffix in suffixes.items():
                if suffix.get(var) is not None:
                    soln[var.name][suffix_name] = suffix.get(var)
        for con in model.component_data_objects(Constraint):
            soln[con.name] = {}
            con_value = con(exception=False)
            soln[con.name]['value'] = con_value
            for suffix_name, suffix in suffixes.items():
                if suffix.get(con) is not None:
                    soln[con.name][suffix_name] = suffix.get(con)
        for obj in model.component_data_objects(Objective):
            soln[obj.name] = {}
            obj_value = obj(exception=False)
            soln[obj.name]['value'] = obj_value
            for suffix_name, suffix in suffixes.items():
                if suffix.get(obj) is not None:
                    soln[obj.name][suffix_name] = suffix.get(obj)
        json.dump(soln, f, indent=2, sort_keys=True)