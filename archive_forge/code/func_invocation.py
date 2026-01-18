import inspect
import logging
import os
import uuid
from functools import wraps
from threading import Lock
import ray._private.signature
from ray import Language, cross_language
from ray._private import ray_option_utils
from ray._private.auto_init_hook import wrap_auto_init
from ray._private.client_mode_hook import (
from ray._private.ray_option_utils import _warn_if_using_deprecated_placement_group
from ray._private.serialization import pickle_dumps
from ray._private.utils import get_runtime_env_info, parse_runtime_env
from ray._raylet import (
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.placement_group import _configure_placement_group_based_on_context
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tracing.tracing_helper import (
def invocation(args, kwargs):
    if self._is_cross_language:
        list_args = cross_language._format_args(worker, args, kwargs)
    elif not args and (not kwargs) and (not self._function_signature):
        list_args = []
    else:
        list_args = ray._private.signature.flatten_args(self._function_signature, args, kwargs)
    if worker.mode == ray._private.worker.LOCAL_MODE:
        assert not self._is_cross_language, 'Cross language remote function cannot be executed locally.'
    object_refs = worker.core_worker.submit_task(self._language, self._function_descriptor, list_args, name if name is not None else '', num_returns, resources, max_retries, retry_exceptions, retry_exception_allowlist, scheduling_strategy, worker.debugger_breakpoint, serialized_runtime_env_info or '{}', generator_backpressure_num_objects)
    worker.debugger_breakpoint = b''
    if num_returns == STREAMING_GENERATOR_RETURN:
        assert len(object_refs) == 1
        generator_ref = object_refs[0]
        return ObjectRefGenerator(generator_ref, worker)
    if len(object_refs) == 1:
        return object_refs[0]
    elif len(object_refs) > 1:
        return object_refs