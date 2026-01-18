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
def createConstraints(self):
    """
        Create the basis constraint y = b(w) (equation 3) and the
        surrogate model constraint y = r_k(w) (equation 5)

        Both constraints are immediately deactivated after creation and
        are activated later as necessary.
        """
    b = self.data

    @b.Constraint(b.ef_outputs.index_set())
    def basis_constraint(b, i):
        ef_output_var = b.ef_outputs[i]
        return ef_output_var == b.basis_expressions[ef_output_var]
    b.basis_constraint.deactivate()
    b.INPUT_OUTPUT = Set(initialize=((i, j) for i in b.ef_outputs.index_set() for j in range(len(b.ef_inputs[i]))))
    b.basis_model_output = Param(b.ef_outputs.index_set(), mutable=True)
    b.grad_basis_model_output = Param(b.INPUT_OUTPUT, mutable=True)
    b.truth_model_output = Param(b.ef_outputs.index_set(), mutable=True)
    b.grad_truth_model_output = Param(b.INPUT_OUTPUT, mutable=True)
    b.value_of_ef_inputs = Param(b.INPUT_OUTPUT, mutable=True)

    @b.Constraint(b.ef_outputs.index_set())
    def sm_constraint_basis(b, i):
        ef_output_var = b.ef_outputs[i]
        return ef_output_var == b.basis_expressions[ef_output_var] + b.truth_model_output[i] - b.basis_model_output[i] + sum(((b.grad_truth_model_output[i, j] - b.grad_basis_model_output[i, j]) * (w - b.value_of_ef_inputs[i, j]) for j, w in enumerate(b.ef_inputs[i])))
    b.sm_constraint_basis.deactivate()