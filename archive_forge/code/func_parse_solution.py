import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def parse_solution(proto: solution_pb2.SolutionProto, mod: model.Model) -> Solution:
    """Returns a Solution equivalent to the input proto."""
    result = Solution()
    if proto.HasField('primal_solution'):
        result.primal_solution = parse_primal_solution(proto.primal_solution, mod)
    if proto.HasField('dual_solution'):
        result.dual_solution = parse_dual_solution(proto.dual_solution, mod)
    result.basis = parse_basis(proto.basis, mod) if proto.HasField('basis') else None
    return result