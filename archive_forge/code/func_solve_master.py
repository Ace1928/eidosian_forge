from pyomo.core.base import (
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR, enforce_dr_degree
def solve_master(model_data, config):
    """
    Solve the master problem
    """
    master_soln = MasterResult()
    if model_data.iteration > 0:
        results = solve_master_feasibility_problem(model_data, config)
        master_soln.feasibility_problem_results = results
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                master_soln.master_model = model_data.master_model
                master_soln.nominal_block = model_data.master_model.scenarios[0, 0]
                master_soln.results = SolverResults()
                setattr(master_soln.results.solver, TIC_TOC_SOLVE_TIME_ATTR, 0)
                master_soln.pyros_termination_condition = pyrosTerminationCondition.time_out
                master_soln.master_subsolver_results = (None, pyrosTerminationCondition.time_out)
                return master_soln
    solver = config.global_solver if config.solve_master_globally else config.local_solver
    return solver_call_master(model_data=model_data, config=config, solver=solver, solve_data=master_soln)