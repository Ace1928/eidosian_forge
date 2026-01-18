import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def parse_solve_result(proto: result_pb2.SolveResultProto, mod: model.Model) -> SolveResult:
    """Returns a SolveResult equivalent to the input proto."""
    result = SolveResult()
    result.termination = parse_termination(_upgrade_termination(proto))
    result.solve_stats = parse_solve_stats(proto.solve_stats)
    for solution_proto in proto.solutions:
        result.solutions.append(solution.parse_solution(solution_proto, mod))
    for primal_ray_proto in proto.primal_rays:
        result.primal_rays.append(solution.parse_primal_ray(primal_ray_proto, mod))
    for dual_ray_proto in proto.dual_rays:
        result.dual_rays.append(solution.parse_dual_ray(dual_ray_proto, mod))
    if proto.HasField('gscip_output'):
        result.gscip_specific_output = proto.gscip_output
    elif proto.HasField('osqp_output'):
        result.osqp_specific_output = proto.osqp_output
    elif proto.HasField('pdlp_output'):
        result.pdlp_specific_output = proto.pdlp_output
    return result