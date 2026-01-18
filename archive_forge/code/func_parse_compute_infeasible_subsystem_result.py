import dataclasses
from typing import Mapping
import immutabledict
from ortools.math_opt import infeasible_subsystem_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import result
def parse_compute_infeasible_subsystem_result(infeasible_system_result: infeasible_subsystem_pb2.ComputeInfeasibleSubsystemResultProto, mod: model.Model) -> ComputeInfeasibleSubsystemResult:
    """Returns an equivalent `ComputeInfeasibleSubsystemResult` to the input proto."""
    return ComputeInfeasibleSubsystemResult(feasibility=result.FeasibilityStatus(infeasible_system_result.feasibility), infeasible_subsystem=parse_model_subset(infeasible_system_result.infeasible_subsystem, mod), is_minimal=infeasible_system_result.is_minimal)