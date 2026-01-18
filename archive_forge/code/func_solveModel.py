import logging
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.trustregion.util import minIgnoreNone, maxIgnoreNone
from pyomo.core import (
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, check_optimal_termination
def solveModel(self):
    """
        Call the specified solver to solve the problem.

        Returns
        -------
            self.data.objs[0] : Current objective value
            step_norm         : Current step size inf norm
            feasibility       : Current feasibility measure

        This also caches the previous values of the vars, just in case
        we need to access them later if a step is rejected
        """
    current_decision_values = self.getCurrentDecisionVariableValues()
    self.data.previous_model_state = self.getCurrentModelState()
    results = self.solver.solve(self.model, keepfiles=self.config.keepfiles, tee=self.config.tee)
    if not check_optimal_termination(results):
        raise ArithmeticError('EXIT: Model solve failed with status {} and termination condition(s) {}.'.format(str(results.solver.status), str(results.solver.termination_condition)))
    self.model.solutions.load_from(results)
    new_decision_values = self.getCurrentDecisionVariableValues()
    step_norm = self.calculateStepSizeInfNorm(current_decision_values, new_decision_values)
    feasibility = self.calculateFeasibility()
    return (self.data.objs[0](), step_norm, feasibility)