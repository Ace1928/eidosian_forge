import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def parse_termination(termination_proto: result_pb2.TerminationProto) -> Termination:
    """Returns a Termination that is equivalent to termination_proto."""
    reason_proto = termination_proto.reason
    limit_proto = termination_proto.limit
    if reason_proto == result_pb2.TERMINATION_REASON_UNSPECIFIED:
        raise ValueError('Termination reason should not be UNSPECIFIED')
    reason_is_limit = reason_proto == result_pb2.TERMINATION_REASON_NO_SOLUTION_FOUND or reason_proto == result_pb2.TERMINATION_REASON_FEASIBLE
    limit_set = limit_proto != result_pb2.LIMIT_UNSPECIFIED
    if reason_is_limit != limit_set:
        raise ValueError(f'Termination limit (={limit_proto})) should take value other than UNSPECIFIED if and only if termination reason (={reason_proto}) is FEASIBLE or NO_SOLUTION_FOUND')
    termination = Termination()
    termination.reason = TerminationReason(reason_proto)
    termination.limit = Limit(limit_proto) if limit_set else None
    termination.detail = termination_proto.detail
    termination.problem_status = parse_problem_status(termination_proto.problem_status)
    termination.objective_bounds = parse_objective_bounds(termination_proto.objective_bounds)
    return termination