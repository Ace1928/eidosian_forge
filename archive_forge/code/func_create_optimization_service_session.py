import json
from typing import List, Optional, Tuple
from google.protobuf import json_format
import requests
from ortools.service.v1 import optimization_pb2
from ortools.math_opt import rpc_pb2
from ortools.math_opt.python import mathopt
from ortools.math_opt.python.ipc import proto_converter
def create_optimization_service_session(api_key: str, deadline_sec: float) -> requests.Session:
    """Creates a session with the appropriate headers.

    This function sets headers for authentication via an API key, and it sets
    deadlines set for the server and the connection.

    Args:
      api_key: Key to the OR API.
      deadline_sec: The number of seconds before the request times out.

    Returns:
      requests.Session a session with the necessary headers to call the
      optimization serive.
    """
    session = requests.Session()
    server_timeout = deadline_sec * (1 - _RELATIVE_TIME_BUFFER)
    session.headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive', 'Keep-Alive': f'timeout={deadline_sec}, max=1', 'X-Server-Timeout': f'{server_timeout}', 'X-Goog-Api-Key': api_key}
    return session