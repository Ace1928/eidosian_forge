import types
from typing import Callable, Optional
from ortools.math_opt import parameters_pb2
from ortools.math_opt.core.python import solver
from ortools.math_opt.python import callback
from ortools.math_opt.python import compute_infeasible_subsystem_result
from ortools.math_opt.python import message_callback
from ortools.math_opt.python import model
from ortools.math_opt.python import model_parameters
from ortools.math_opt.python import parameters
from ortools.math_opt.python import result
from pybind11_abseil.status import StatusNotOk
def compute_infeasible_subsystem(opt_model: model.Model, solver_type: parameters.SolverType, *, params: Optional[parameters.SolveParameters]=None, msg_cb: Optional[message_callback.SolveMessageCallback]=None) -> compute_infeasible_subsystem_result.ComputeInfeasibleSubsystemResult:
    """Computes an infeasible subsystem of the input model.

    Args:
      opt_model: The optimization model to check for infeasibility.
      solver_type: Which solver to use to compute the infeasible subsystem. As of
        August 2023, the only supported solver is Gurobi.
      params: Configuration of the underlying solver.
      msg_cb: A callback that gives back the underlying solver's logs by the line.

    Returns:
      An `ComputeInfeasibleSubsystemResult` where `feasibility` indicates if the
      problem was proven infeasible.

    Throws:
      RuntimeError: on invalid inputs or an internal solver error.
    """
    params = params or parameters.SolveParameters()
    model_proto = opt_model.export_model()
    try:
        proto_result = solver.compute_infeasible_subsystem(model_proto, solver_type.value, parameters_pb2.SolverInitializerProto(), params.to_proto(), msg_cb, None)
    except StatusNotOk as e:
        raise RuntimeError(str(e)) from None
    return compute_infeasible_subsystem_result.parse_compute_infeasible_subsystem_result(proto_result, opt_model)