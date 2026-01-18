import dataclasses
from typing import Mapping
import immutabledict
from ortools.math_opt import infeasible_subsystem_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import result
def parse_model_subset(model_subset: infeasible_subsystem_pb2.ModelSubsetProto, mod: model.Model) -> ModelSubset:
    """Returns an equivalent `ModelSubset` to the input proto."""
    if model_subset.quadratic_constraints:
        raise NotImplementedError('quadratic_constraints not yet implemented for ModelSubset in Python')
    if model_subset.second_order_cone_constraints:
        raise NotImplementedError('second_order_cone_constraints not yet implemented for ModelSubset in Python')
    if model_subset.sos1_constraints:
        raise NotImplementedError('sos1_constraints not yet implemented for ModelSubset in Python')
    if model_subset.sos2_constraints:
        raise NotImplementedError('sos2_constraints not yet implemented for ModelSubset in Python')
    if model_subset.indicator_constraints:
        raise NotImplementedError('indicator_constraints not yet implemented for ModelSubset in Python')
    return ModelSubset(variable_bounds={mod.get_variable(var_id): parse_model_subset_bounds(bounds) for var_id, bounds in model_subset.variable_bounds.items()}, variable_integrality=frozenset((mod.get_variable(var_id) for var_id in model_subset.variable_integrality)), linear_constraints={mod.get_linear_constraint(con_id): parse_model_subset_bounds(bounds) for con_id, bounds in model_subset.linear_constraints.items()})