import logging
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.core import Block, Constraint, Var
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction
def build_model_size_report(model):
    """Build a model size report object."""
    report = ModelSizeReport()
    activated_disjunctions = ComponentSet()
    activated_disjuncts = ComponentSet()
    fixed_true_disjuncts = ComponentSet()
    activated_constraints = ComponentSet()
    activated_vars = ComponentSet()
    new_containers = (model,)
    while new_containers:
        new_activated_disjunctions = ComponentSet()
        new_activated_disjuncts = ComponentSet()
        new_fixed_true_disjuncts = ComponentSet()
        new_activated_constraints = ComponentSet()
        for container in new_containers:
            next_activated_disjunctions, next_fixed_true_disjuncts, next_activated_disjuncts, next_activated_constraints = _process_activated_container(container)
            new_activated_disjunctions.update(next_activated_disjunctions)
            new_activated_disjuncts.update(next_activated_disjuncts)
            new_fixed_true_disjuncts.update(next_fixed_true_disjuncts)
            new_activated_constraints.update(next_activated_constraints)
        new_containers = new_activated_disjuncts - activated_disjuncts | new_fixed_true_disjuncts - fixed_true_disjuncts
        activated_disjunctions.update(new_activated_disjunctions)
        activated_disjuncts.update(new_activated_disjuncts)
        fixed_true_disjuncts.update(new_fixed_true_disjuncts)
        activated_constraints.update(new_activated_constraints)
    activated_vars.update((var for constr in activated_constraints for var in EXPR.identify_variables(constr.body, include_fixed=False)))
    activated_vars.update((disj.indicator_var.get_associated_binary() for disj in activated_disjuncts))
    report.activated = Bunch()
    report.activated.variables = len(activated_vars)
    report.activated.binary_variables = sum((1 for v in activated_vars if v.is_binary()))
    report.activated.integer_variables = sum((1 for v in activated_vars if v.is_integer() and (not v.is_binary())))
    report.activated.continuous_variables = sum((1 for v in activated_vars if v.is_continuous()))
    report.activated.disjunctions = len(activated_disjunctions)
    report.activated.disjuncts = len(activated_disjuncts)
    report.activated.constraints = len(activated_constraints)
    report.activated.nonlinear_constraints = sum((1 for c in activated_constraints if c.body.polynomial_degree() not in (1, 0)))
    report.overall = Bunch()
    block_like = (Block, Disjunct)
    all_vars = ComponentSet(model.component_data_objects(Var, descend_into=block_like))
    report.overall.variables = len(all_vars)
    report.overall.binary_variables = sum((1 for v in all_vars if v.is_binary()))
    report.overall.integer_variables = sum((1 for v in all_vars if v.is_integer() and (not v.is_binary())))
    report.overall.continuous_variables = sum((1 for v in all_vars if v.is_continuous()))
    report.overall.disjunctions = sum((1 for d in model.component_data_objects(Disjunction, descend_into=block_like)))
    report.overall.disjuncts = sum((1 for d in model.component_data_objects(Disjunct, descend_into=block_like)))
    report.overall.constraints = sum((1 for c in model.component_data_objects(Constraint, descend_into=block_like)))
    report.overall.nonlinear_constraints = sum((1 for c in model.component_data_objects(Constraint, descend_into=block_like) if c.body.polynomial_degree() not in (1, 0)))
    report.warning = Bunch()
    report.warning.unassociated_disjuncts = sum((1 for d in model.component_data_objects(Disjunct, descend_into=block_like) if not d.indicator_var.fixed and d not in activated_disjuncts))
    return report