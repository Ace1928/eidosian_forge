from math import copysign
from pyomo.core import minimize, value
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
def add_oa_cuts(target_model, dual_values, jacobians, objective_sense, mip_constraint_polynomial_degree, mip_iter, config, timing, cb_opt=None, linearize_active=True, linearize_violated=True):
    """Adds OA cuts.

    Generates and adds OA cuts (linearizes nonlinear constraints).
    For nonconvex problems, turn on 'config.add_slack'.
    Slack variables will always be used for nonlinear equality constraints.

    Parameters
    ----------
    target_model : Pyomo model
        The relaxed linear model.
    dual_values : list
        The value of the duals for each constraint.
    jacobians : ComponentMap
        Map nonlinear_constraint --> Map(variable --> jacobian of constraint w.r.t. variable).
    objective_sense : Int
        Objective sense of model.
    mip_constraint_polynomial_degree : Set
        The polynomial degrees of constraints that are regarded as linear.
    mip_iter : Int
        MIP iteration counter.
    config : ConfigBlock
        The specific configurations for MindtPy.
    cb_opt : SolverFactory, optional
        Gurobi_persistent solver, by default None.
    linearize_active : bool, optional
        Whether to linearize the active nonlinear constraints, by default True.
    linearize_violated : bool, optional
        Whether to linearize the violated nonlinear constraints, by default True.
    """
    with time_code(timing, 'OA cut generation'):
        for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
            if constr.body.polynomial_degree() in mip_constraint_polynomial_degree:
                continue
            constr_vars = list(EXPR.identify_variables(constr.body))
            jacs = jacobians
            if constr.has_ub() and constr.has_lb() and (value(constr.lower) == value(constr.upper)) and config.equality_relaxation:
                sign_adjust = -1 if objective_sense == minimize else 1
                rhs = constr.lower
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                target_model.MindtPy_utils.cuts.oa_cuts.add(expr=copysign(1, sign_adjust * dual_values[index]) * (sum((value(jacs[constr][var]) * (var - value(var)) for var in EXPR.identify_variables(constr.body))) + value(constr.body) - rhs) - (slack_var if config.add_slack else 0) <= 0)
                if config.single_tree and config.mip_solver == 'gurobi_persistent' and (mip_iter > 0) and (cb_opt is not None):
                    cb_opt.cbLazy(target_model.MindtPy_utils.cuts.oa_cuts[len(target_model.MindtPy_utils.cuts.oa_cuts)])
            else:
                if (constr.has_ub() and (linearize_active and abs(constr.uslack()) < config.zero_tolerance) or (linearize_violated and constr.uslack() < 0) or (config.linearize_inactive and constr.uslack() > 0)) or ('MindtPy_utils.objective_constr' in constr.name and constr.has_ub()):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                    target_model.MindtPy_utils.cuts.oa_cuts.add(expr=sum((value(jacs[constr][var]) * (var - var.value) for var in constr_vars)) + value(constr.body) - (slack_var if config.add_slack else 0) <= value(constr.upper))
                    if config.single_tree and config.mip_solver == 'gurobi_persistent' and (mip_iter > 0) and (cb_opt is not None):
                        cb_opt.cbLazy(target_model.MindtPy_utils.cuts.oa_cuts[len(target_model.MindtPy_utils.cuts.oa_cuts)])
                if (constr.has_lb() and (linearize_active and abs(constr.lslack()) < config.zero_tolerance) or (linearize_violated and constr.lslack() < 0) or (config.linearize_inactive and constr.lslack() > 0)) or ('MindtPy_utils.objective_constr' in constr.name and constr.has_lb()):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                    target_model.MindtPy_utils.cuts.oa_cuts.add(expr=sum((value(jacs[constr][var]) * (var - var.value) for var in constr_vars)) + value(constr.body) + (slack_var if config.add_slack else 0) >= value(constr.lower))
                    if config.single_tree and config.mip_solver == 'gurobi_persistent' and (mip_iter > 0) and (cb_opt is not None):
                        cb_opt.cbLazy(target_model.MindtPy_utils.cuts.oa_cuts[len(target_model.MindtPy_utils.cuts.oa_cuts)])