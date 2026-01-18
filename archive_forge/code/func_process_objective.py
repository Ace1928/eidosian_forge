import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def process_objective(self, update_var_con_list=True):
    """Process model objective function.

        Check that the model has only 1 valid objective.
        If the objective is nonlinear, move it into the constraints.
        If no objective function exists, emit a warning and create a dummy objective.

        Parameters
        ----------
        update_var_con_list : bool, optional
            Whether to update the variable/constraint/objective lists, by default True.
            Currently, update_var_con_list will be set to False only when add_regularization is not None in MindtPy.
        """
    config = self.config
    m = self.working_model
    util_block = getattr(m, self.util_block_name)
    active_objectives = list(m.component_data_objects(ctype=Objective, active=True, descend_into=True))
    self.results.problem.number_of_objectives = len(active_objectives)
    if len(active_objectives) == 0:
        config.logger.warning('Model has no active objectives. Adding dummy objective.')
        util_block.dummy_objective = Objective(expr=1)
        main_obj = util_block.dummy_objective
    elif len(active_objectives) > 1:
        raise ValueError('Model has multiple active objectives.')
    else:
        main_obj = active_objectives[0]
    self.results.problem.sense = ProblemSense.minimize if main_obj.sense == 1 else ProblemSense.maximize
    self.objective_sense = main_obj.sense
    if main_obj.expr.polynomial_degree() not in self.mip_objective_polynomial_degree or config.move_objective:
        if config.move_objective:
            config.logger.info('Moving objective to constraint set.')
        else:
            config.logger.info('Objective is nonlinear. Moving it to constraint set.')
        util_block.objective_value = VarList(domain=Reals, initialize=0)
        util_block.objective_constr = ConstraintList()
        if main_obj.expr.polynomial_degree() not in self.mip_objective_polynomial_degree and config.partition_obj_nonlinear_terms and (main_obj.expr.__class__ is EXPR.SumExpression):
            repn = generate_standard_repn(main_obj.expr, quadratic=2 in self.mip_objective_polynomial_degree)
            linear_subexpr = repn.constant + sum((coef * var for coef, var in zip(repn.linear_coefs, repn.linear_vars))) + sum((coef * var1 * var2 for coef, (var1, var2) in zip(repn.quadratic_coefs, repn.quadratic_vars)))
            epigraph_reformulation(linear_subexpr, util_block.objective_value, util_block.objective_constr, config.use_mcpp, main_obj.sense)
            nonlinear_subexpr = repn.nonlinear_expr
            if nonlinear_subexpr.__class__ is EXPR.SumExpression:
                for subsubexpr in nonlinear_subexpr.args:
                    epigraph_reformulation(subsubexpr, util_block.objective_value, util_block.objective_constr, config.use_mcpp, main_obj.sense)
            else:
                epigraph_reformulation(nonlinear_subexpr, util_block.objective_value, util_block.objective_constr, config.use_mcpp, main_obj.sense)
        else:
            epigraph_reformulation(main_obj.expr, util_block.objective_value, util_block.objective_constr, config.use_mcpp, main_obj.sense)
        main_obj.deactivate()
        util_block.objective = Objective(expr=sum(util_block.objective_value[:]), sense=main_obj.sense)
        if main_obj.expr.polynomial_degree() not in self.mip_objective_polynomial_degree or (config.move_objective and update_var_con_list):
            util_block.variable_list.extend(util_block.objective_value[:])
            util_block.continuous_variable_list.extend(util_block.objective_value[:])
            util_block.constraint_list.extend(util_block.objective_constr[:])
            util_block.objective_list.append(util_block.objective)
            for constr in util_block.objective_constr[:]:
                if constr.body.polynomial_degree() in self.mip_constraint_polynomial_degree:
                    util_block.linear_constraint_list.append(constr)
                else:
                    util_block.nonlinear_constraint_list.append(constr)