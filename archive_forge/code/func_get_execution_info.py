import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def get_execution_info(self, job_id, function_descriptor):
    """Get the FunctionExecutionInfo of a remote function.
        Args:
            job_id: ID of the job that the function belongs to.
            function_descriptor: The FunctionDescriptor of the function to get.
        Returns:
            A FunctionExecutionInfo object.
        """
    function_id = function_descriptor.function_id
    if function_id in self._function_execution_info:
        return self._function_execution_info[function_id]
    if self._worker.load_code_from_local:
        if not function_descriptor.is_actor_method():
            if self._load_function_from_local(function_descriptor) is True:
                return self._function_execution_info[function_id]
    with profiling.profile('wait_for_function'):
        self._wait_for_function(function_descriptor, job_id)
    try:
        function_id = function_descriptor.function_id
        info = self._function_execution_info[function_id]
    except KeyError as e:
        message = 'Error occurs in get_execution_info: job_id: %s, function_descriptor: %s. Message: %s' % (job_id, function_descriptor, e)
        raise KeyError(message)
    return info