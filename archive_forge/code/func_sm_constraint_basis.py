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
@b.Constraint(b.ef_outputs.index_set())
def sm_constraint_basis(b, i):
    ef_output_var = b.ef_outputs[i]
    return ef_output_var == b.basis_expressions[ef_output_var] + b.truth_model_output[i] - b.basis_model_output[i] + sum(((b.grad_truth_model_output[i, j] - b.grad_basis_model_output[i, j]) * (w - b.value_of_ef_inputs[i, j]) for j, w in enumerate(b.ef_inputs[i])))