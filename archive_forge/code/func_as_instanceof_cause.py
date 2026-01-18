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
def as_instanceof_cause(self):
    """Returns an exception that is an instance of the cause's class.

        The returned exception will inherit from both RayTaskError and the
        cause class and will contain all of the attributes of the cause
        exception.
        """
    cause_cls = self.cause.__class__
    if issubclass(RayTaskError, cause_cls):
        return self
    error_msg = str(self)

    class cls(RayTaskError, cause_cls):

        def __init__(self, cause):
            self.cause = cause
            self.args = (cause,)

        def __getattr__(self, name):
            return getattr(self.cause, name)

        def __str__(self):
            return error_msg
    name = f'RayTaskError({cause_cls.__name__})'
    cls.__name__ = name
    cls.__qualname__ = name
    return cls(self.cause)