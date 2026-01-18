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
def add_lazy_oa_cuts(self, target_model, dual_values, mindtpy_solver, config, opt, linearize_active=True, linearize_violated=True):
    """Linearizes nonlinear constraints; add the OA cuts through CPLEX inherent function self.add()
        For nonconvex problems, turn on 'config.add_slack'. Slack variables will always be used for
        nonlinear equality constraints.

        Parameters
        ----------
        target_model : Pyomo model
            The MIP main problem.
        dual_values : list
            The value of the duals for each constraint.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        linearize_active : bool, optional
            Whether to linearize the active nonlinear constraints, by default True.
        linearize_violated : bool, optional
            Whether to linearize the violated nonlinear constraints, by default True.
        """
    config.logger.debug('Adding OA cuts')
    with time_code(mindtpy_solver.timing, 'OA cut generation'):
        for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
            if constr.body.polynomial_degree() in mindtpy_solver.mip_constraint_polynomial_degree:
                continue
            constr_vars = list(identify_variables(constr.body))
            jacs = mindtpy_solver.jacobians
            if constr.has_ub() and constr.has_lb() and (value(constr.lower) == value(constr.upper)):
                sign_adjust = -1 if mindtpy_solver.objective_sense == minimize else 1
                rhs = constr.lower
                pyomo_expr = copysign(1, sign_adjust * dual_values[index]) * (sum((value(jacs[constr][var]) * (var - value(var)) for var in EXPR.identify_variables(constr.body))) + value(constr.body) - rhs)
                cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients), sense='L', rhs=cplex_rhs)
                if self.get_solution_source() == cplex.callbacks.SolutionSource.mipstart_solution:
                    mindtpy_solver.mip_start_lazy_oa_cuts.append([cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients), 'L', cplex_rhs])
            else:
                if (constr.has_ub() and (linearize_active and abs(constr.uslack()) < config.zero_tolerance) or (linearize_violated and constr.uslack() < 0) or (config.linearize_inactive and constr.uslack() > 0)) or ('MindtPy_utils.objective_constr' in constr.name and constr.has_ub()):
                    pyomo_expr = sum((value(jacs[constr][var]) * (var - var.value) for var in constr_vars)) + value(constr.body)
                    cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients), sense='L', rhs=value(constr.upper) + cplex_rhs)
                    if self.get_solution_source() == cplex.callbacks.SolutionSource.mipstart_solution:
                        mindtpy_solver.mip_start_lazy_oa_cuts.append([cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients), 'L', value(constr.upper) + cplex_rhs])
                if (constr.has_lb() and (linearize_active and abs(constr.lslack()) < config.zero_tolerance) or (linearize_violated and constr.lslack() < 0) or (config.linearize_inactive and constr.lslack() > 0)) or ('MindtPy_utils.objective_constr' in constr.name and constr.has_lb()):
                    pyomo_expr = sum((value(jacs[constr][var]) * (var - self.get_values(opt._pyomo_var_to_solver_var_map[var])) for var in constr_vars)) + value(constr.body)
                    cplex_rhs = -generate_standard_repn(pyomo_expr).constant
                    cplex_expr, _ = opt._get_expr_from_pyomo_expr(pyomo_expr)
                    self.add(constraint=cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients), sense='G', rhs=value(constr.lower) + cplex_rhs)
                    if self.get_solution_source() == cplex.callbacks.SolutionSource.mipstart_solution:
                        mindtpy_solver.mip_start_lazy_oa_cuts.append([cplex.SparsePair(ind=cplex_expr.variables, val=cplex_expr.coefficients), 'G', value(constr.lower) + cplex_rhs])