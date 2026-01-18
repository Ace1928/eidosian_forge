from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_no_good_cuts
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from math import copysign
from pyomo.contrib.mindtpy.util import (
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.opt import TerminationCondition as tc
from pyomo.core import minimize, value
from pyomo.core.expr import identify_variables
def copy_lazy_var_list_values(self, opt, from_list, to_list, config, skip_stale=False, skip_fixed=True):
    """This function copies variable values from one list to another.
        Rounds to Binary/Integer if necessary.
        Sets to zero for NonNegativeReals if necessary.

        Parameters
        ----------
        opt : SolverFactory
            The cplex_persistent solver.
        from_list : list
            The variable list that provides the values to copy from.
        to_list : list
            The variable list that needs to set value.
        config : ConfigBlock
            The specific configurations for MindtPy.
        skip_stale : bool, optional
            Whether to skip the stale variables, by default False.
        skip_fixed : bool, optional
            Whether to skip the fixed variables, by default True.
        """
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue
        if skip_fixed and v_to.is_fixed():
            continue
        v_val = self.get_values(opt._pyomo_var_to_solver_var_map[v_from])
        set_var_valid_value(v_to, v_val, config.integer_tolerance, config.zero_tolerance, ignore_integrality=False)