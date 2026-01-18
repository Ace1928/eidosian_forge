from google.protobuf import message
from ortools.service.v1 import optimization_pb2
from ortools.math_opt import rpc_pb2
from ortools.math_opt.python import normalize
def convert_response(api_response: optimization_pb2.SolveMathOptModelResponse) -> rpc_pb2.SolveResponse:
    """Converts a `SolveMathOptModelResponse` to a `SolveResponse`.

    Args:
      api_response: A `SolveMathOptModelResponse` response built from a MathOpt
        model.

    Returns:
      A `SolveResponse` response built from a MathOpt model.
    """
    api_response.DiscardUnknownFields()
    normalize.math_opt_normalize_proto(api_response)
    response = rpc_pb2.SolveResponse.FromString(api_response.SerializeToString())
    response.DiscardUnknownFields()
    return response