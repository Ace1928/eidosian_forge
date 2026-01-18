from math import fabs
from pyomo.common.collections import ComponentSet
from pyomo.core import TransformationFactory, value, Constraint, Block
def add_no_good_cut(target_model_util_block, config):
    """Cut the current integer solution from the target model."""
    var_value_is_one = ComponentSet()
    var_value_is_zero = ComponentSet()
    for var in target_model_util_block.transformed_boolean_variable_list:
        _record_binary_value(var, var_value_is_one, var_value_is_zero, config.integer_tolerance)
    disjuncts = []
    if config.force_subproblem_nlp:
        for var in target_model_util_block.discrete_variable_list:
            if var.is_binary():
                _record_binary_value(var, var_value_is_one, var_value_is_zero, config.integer_tolerance)
            else:
                val = round(value(var), 0)
                less = var <= val - 1
                more = var >= val + 1
                disjuncts.extend([less, more])
    assert (var_value_is_one or var_value_is_zero) or len(disjuncts) == 0
    int_cut = sum((1 - v for v in var_value_is_one)) + sum((v for v in var_value_is_zero)) >= 1
    if len(disjuncts) > 0:
        idx = len(target_model_util_block.no_good_disjunctions)
        target_model_util_block.no_good_disjunctions[idx] = [[disj] for disj in disjuncts] + [[int_cut]]
        config.logger.debug('Adding no-good disjunction: %s' % _disjunction_to_str(target_model_util_block.no_good_disjunctions[idx]))
        TransformationFactory(config.discrete_problem_transformation).apply_to(target_model_util_block, targets=[target_model_util_block.no_good_disjunctions[idx]])
    else:
        config.logger.debug('Adding no-good cut: %s' % int_cut)
        target_model_util_block.no_good_cuts.add(expr=int_cut)