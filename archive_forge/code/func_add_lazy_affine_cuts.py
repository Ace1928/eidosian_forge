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
def add_lazy_affine_cuts(self, mindtpy_solver, config, opt):
    """Adds affine cuts using MCPP.

        Add affine cuts through CPLEX inherent function self.add().

        Parameters
        ----------
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        """
    with time_code(mindtpy_solver.timing, 'Affine cut generation'):
        m = mindtpy_solver.mip
        config.logger.debug('Adding affine cuts')
        counter = 0
        for constr in m.MindtPy_utils.nonlinear_constraint_list:
            vars_in_constr = list(identify_variables(constr.body))
            if any((var.value is None for var in vars_in_constr)):
                continue
            try:
                mc_eqn = mc(constr.body)
            except MCPP_Error as e:
                config.logger.error(e, exc_info=True)
                config.logger.debug('Skipping constraint %s due to MCPP error' % constr.name)
                continue
            ccSlope = mc_eqn.subcc()
            cvSlope = mc_eqn.subcv()
            ccStart = mc_eqn.concave()
            cvStart = mc_eqn.convex()
            concave_cut_valid = True
            convex_cut_valid = True
            for var in vars_in_constr:
                if not var.fixed:
                    if ccSlope[var] == float('nan') or ccSlope[var] == float('inf'):
                        concave_cut_valid = False
                    if cvSlope[var] == float('nan') or cvSlope[var] == float('inf'):
                        convex_cut_valid = False
            if ccStart == float('nan') or ccStart == float('inf'):
                concave_cut_valid = False
            if cvStart == float('nan') or cvStart == float('inf'):
                convex_cut_valid = False
            if not any(ccSlope.values()):
                concave_cut_valid = False
            if not any(cvSlope.values()):
                convex_cut_valid = False
            if not (concave_cut_valid or convex_cut_valid):
                continue
            ub_int = min(value(constr.upper), mc_eqn.upper()) if constr.has_ub() else mc_eqn.upper()
            lb_int = max(value(constr.lower), mc_eqn.lower()) if constr.has_lb() else mc_eqn.lower()
            if concave_cut_valid:
                pyomo_concave_cut = sum((ccSlope[var] * (var - var.value) for var in vars_in_constr if not var.fixed)) + ccStart
                cplex_concave_rhs = generate_standard_repn(pyomo_concave_cut).constant
                cplex_concave_cut, _ = opt._get_expr_from_pyomo_expr(pyomo_concave_cut)
                self.add(constraint=cplex.SparsePair(ind=cplex_concave_cut.variables, val=cplex_concave_cut.coefficients), sense='G', rhs=lb_int - cplex_concave_rhs)
                counter += 1
            if convex_cut_valid:
                pyomo_convex_cut = sum((cvSlope[var] * (var - var.value) for var in vars_in_constr if not var.fixed)) + cvStart
                cplex_convex_rhs = generate_standard_repn(pyomo_convex_cut).constant
                cplex_convex_cut, _ = opt._get_expr_from_pyomo_expr(pyomo_convex_cut)
                self.add(constraint=cplex.SparsePair(ind=cplex_convex_cut.variables, val=cplex_convex_cut.coefficients), sense='L', rhs=ub_int - cplex_convex_rhs)
                counter += 1
        config.logger.debug('Added %s affine cuts' % counter)