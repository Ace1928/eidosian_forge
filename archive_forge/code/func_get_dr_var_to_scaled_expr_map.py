from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, coefficient_matching
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import _VarData as VarData
from itertools import chain
from pyomo.common.dependencies import numpy as np
def get_dr_var_to_scaled_expr_map(decision_rule_eqns, second_stage_vars, uncertain_params, decision_rule_vars):
    """
    Generate mapping from decision rule variables
    to their terms in a model's DR expression.
    """
    var_to_scaled_expr_map = ComponentMap()
    ssv_dr_eq_zip = zip(second_stage_vars, decision_rule_eqns)
    for ssv_idx, (ssv, dr_eq) in enumerate(ssv_dr_eq_zip):
        for term in dr_eq.body.args:
            is_ssv_term = isinstance(term.args[0], int) and term.args[0] == -1 and isinstance(term.args[1], VarData)
            if not is_ssv_term:
                dr_var = term.args[1]
                var_to_scaled_expr_map[dr_var] = term
    return var_to_scaled_expr_map