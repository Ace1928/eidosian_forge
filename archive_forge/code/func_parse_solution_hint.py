import dataclasses
from typing import Dict, List, Optional
from ortools.math_opt import model_parameters_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.python import sparse_containers
def parse_solution_hint(hint_proto: model_parameters_pb2.SolutionHintProto, mod: model.Model) -> SolutionHint:
    """Returns an equivalent SolutionHint to `hint_proto`.

    Args:
      hint_proto: The solution, as encoded by the ids of the variables and
        constraints.
      mod: A MathOpt Model that must contain variables and linear constraints with
        the ids from hint_proto.

    Returns:
      A SolutionHint equivalent.

    Raises:
      ValueError if hint_proto is invalid or refers to variables or constraints
      not in mod.
    """
    return SolutionHint(variable_values=sparse_containers.parse_variable_map(hint_proto.variable_values, mod), dual_values=sparse_containers.parse_linear_constraint_map(hint_proto.dual_values, mod))