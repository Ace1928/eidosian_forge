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
def replaceExternalFunctionsWithVariables(self):
    """
        This method sets up essential data objects on the new trf_data block
        on the model as well as triggers the replacement of external functions
        in expressions trees.

        Data objects created:
            self.data.all_variables : ComponentSet
                A set of all variables on the model, including "holder"
                variables from the EF replacement
            self.data.truth_models : ComponentMap
                A component map for replaced nodes that keeps track of
                the truth model for that replacement.
            self.data.basis_expressions : ComponentMap
                A component map for the Pyomo expressions for basis functions
                as they apply to each variable
            self.data.ef_inputs : Dict
                A dictionary that tracks the input variables for each EF
            self.data.ef_outputs : VarList
                A list of the "holder" variables which replaced the original
                External Function expressions
        """
    self.data.all_variables = ComponentSet()
    self.data.truth_models = ComponentMap()
    self.data.basis_expressions = ComponentMap()
    self.data.ef_inputs = {}
    self.data.ef_outputs = VarList()
    number_of_equality_constraints = 0
    for con in self.model.component_data_objects(Constraint, active=True):
        if con.lb == con.ub and con.lb is not None:
            number_of_equality_constraints += 1
        self._remove_ef_from_expr(con)
    self.degrees_of_freedom = len(list(self.data.all_variables)) - number_of_equality_constraints
    if self.degrees_of_freedom != len(self.decision_variables):
        raise ValueError('replaceExternalFunctionsWithVariables: The degrees of freedom %d do not match the number of decision variables supplied %d.' % (self.degrees_of_freedom, len(self.decision_variables)))
    for var in self.decision_variables:
        if var not in self.data.all_variables:
            raise ValueError(f'replaceExternalFunctionsWithVariables: The supplied decision variable {var.name} cannot be found in the model variables.')
    self.data.objs = list(self.model.component_data_objects(Objective, active=True))
    for ef in self.model.component_objects(ExternalFunction):
        ef.parent_block().del_component(ef)
    if len(self.data.objs) != 1:
        raise ValueError('replaceExternalFunctionsWithVariables: TrustRegion only supports models with a single active Objective.')
    if self.data.objs[0].sense == maximize:
        self.data.objs[0].expr = -1 * self.data.objs[0].expr
        self.data.objs[0].sense = minimize
    self._remove_ef_from_expr(self.data.objs[0])
    for i in self.data.ef_outputs:
        self.data.ef_inputs[i] = list(identify_variables(self.data.truth_models[self.data.ef_outputs[i]], include_fixed=False))
    self.data.all_variables.update(self.data.ef_outputs.values())
    self.data.all_variables = list(self.data.all_variables)