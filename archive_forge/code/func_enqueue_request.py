import asyncio
import time
from dataclasses import dataclass
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import (
from ray._private.signature import extract_signature, flatten_args, recover_args
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.utils import extract_self_if_method_call
from ray.serve.exceptions import RayServeException
from ray.util.annotations import PublicAPI
def enqueue_request(args, kwargs) -> asyncio.Future:
    self = extract_self_if_method_call(args, _func)
    flattened_args: List = flatten_args(extract_signature(_func), args, kwargs)
    if self is None:
        batch_queue_object = _func
    else:
        batch_queue_object = self
        flattened_args = flattened_args[2:]
    batch_queue = lazy_batch_queue_wrapper.queue
    if hasattr(batch_queue_object, '_ray_serve_max_batch_size'):
        new_max_batch_size = getattr(batch_queue_object, '_ray_serve_max_batch_size')
        _validate_max_batch_size(new_max_batch_size)
        batch_queue.max_batch_size = new_max_batch_size
    if hasattr(batch_queue_object, '_ray_serve_batch_wait_timeout_s'):
        new_batch_wait_timeout_s = getattr(batch_queue_object, '_ray_serve_batch_wait_timeout_s')
        _validate_batch_wait_timeout_s(new_batch_wait_timeout_s)
        batch_queue.batch_wait_timeout_s = new_batch_wait_timeout_s
    future = get_or_create_event_loop().create_future()
    batch_queue.put(_SingleRequest(self, flattened_args, future))
    return future