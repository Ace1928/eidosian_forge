import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def parse_solve_stats(proto: result_pb2.SolveStatsProto) -> SolveStats:
    """Returns an equivalent SolveStats from the input proto."""
    result = SolveStats()
    result.solve_time = proto.solve_time.ToTimedelta()
    result.simplex_iterations = proto.simplex_iterations
    result.barrier_iterations = proto.barrier_iterations
    result.first_order_iterations = proto.first_order_iterations
    result.node_count = proto.node_count
    return result