import os
from traceback import format_exception
from typing import Optional, Union
import colorama
import ray._private.ray_constants as ray_constants
import ray.cloudpickle as pickle
from ray._raylet import ActorID, TaskID, WorkerID
from ray.core.generated.common_pb2 import (
from ray.util.annotations import DeveloperAPI, PublicAPI
import setproctitle
@staticmethod
def from_ray_exception(ray_exception):
    if ray_exception.language == PYTHON:
        try:
            return pickle.loads(ray_exception.serialized_exception)
        except Exception as e:
            msg = 'Failed to unpickle serialized exception'
            raise RuntimeError(msg) from e
    else:
        return CrossLanguageError(ray_exception)