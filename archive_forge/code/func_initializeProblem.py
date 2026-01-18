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
def initializeProblem(self):
    """
        Initializes appropriate constraints, values, etc. for TRF problem

        Returns
        -------
            objective_value : Initial objective
            feasibility     : Initial feasibility measure

        STEPS:
            1. Create and solve PMP (eq. 3) and set equal to "x_0"
            2. Evaluate d(w_0)
            3. Evaluate initial feasibility measure (theta(x_0))
            4. Create initial SM (difference btw. low + high fidelity models)

        """
    self.replaceExternalFunctionsWithVariables()
    self.initial_decision_bounds = {}
    for var in self.decision_variables:
        self.initial_decision_bounds[var.name] = [var.lb, var.ub]
    self.createConstraints()
    self.data.basis_constraint.activate()
    objective_value, _, _ = self.solveModel()
    self.data.basis_constraint.deactivate()
    self.updateSurrogateModel()
    feasibility = self.calculateFeasibility()
    self.data.sm_constraint_basis.activate()
    return (objective_value, feasibility)