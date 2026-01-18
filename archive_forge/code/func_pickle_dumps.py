import io
import logging
import threading
import traceback
from typing import Any
import google.protobuf.message
import ray._private.utils
import ray.cloudpickle as pickle
from ray._private import ray_constants
from ray._raylet import (
from ray.core.generated.common_pb2 import ErrorType, RayErrorInfo
from ray.exceptions import (
from ray.util import serialization_addons
from ray.util import inspect_serializability
def pickle_dumps(obj: Any, error_msg: str):
    """Wrap cloudpickle.dumps to provide better error message
    when the object is not serializable.
    """
    try:
        return pickle.dumps(obj)
    except TypeError as e:
        sio = io.StringIO()
        inspect_serializability(obj, print_file=sio)
        msg = f'{error_msg}:\n{sio.getvalue()}'
        raise TypeError(msg) from e