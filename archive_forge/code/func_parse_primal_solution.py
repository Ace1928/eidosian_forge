import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def parse_primal_solution(proto: solution_pb2.PrimalSolutionProto, mod: model.Model) -> PrimalSolution:
    """Returns an equivalent PrimalSolution from the input proto."""
    result = PrimalSolution()
    result.objective_value = proto.objective_value
    result.variable_values = sparse_containers.parse_variable_map(proto.variable_values, mod)
    status_proto = proto.feasibility_status
    if status_proto == solution_pb2.SOLUTION_STATUS_UNSPECIFIED:
        raise ValueError('Primal solution feasibility status should not be UNSPECIFIED')
    result.feasibility_status = SolutionStatus(status_proto)
    return result