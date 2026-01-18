import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def parse_dual_solution(proto: solution_pb2.DualSolutionProto, mod: model.Model) -> DualSolution:
    """Returns an equivalent DualSolution from the input proto."""
    result = DualSolution()
    result.objective_value = proto.objective_value if proto.HasField('objective_value') else None
    result.dual_values = sparse_containers.parse_linear_constraint_map(proto.dual_values, mod)
    result.reduced_costs = sparse_containers.parse_variable_map(proto.reduced_costs, mod)
    status_proto = proto.feasibility_status
    if status_proto == solution_pb2.SOLUTION_STATUS_UNSPECIFIED:
        raise ValueError('Dual solution feasibility status should not be UNSPECIFIED')
    result.feasibility_status = SolutionStatus(status_proto)
    return result