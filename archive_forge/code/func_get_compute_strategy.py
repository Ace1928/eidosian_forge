import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def get_compute_strategy(fn: 'UserDefinedFunction', fn_constructor_args: Optional[Iterable[Any]]=None, compute: Optional[Union[str, 'ComputeStrategy']]=None, concurrency: Optional[Union[int, Tuple[int, int]]]=None) -> 'ComputeStrategy':
    """Get `ComputeStrategy` based on the function or class, and concurrency
    information.

    Args:
        fn: The function or generator to apply to a record batch, or a class type
            that can be instantiated to create such a callable.
        fn_constructor_args: Positional arguments to pass to ``fn``'s constructor.
        compute: Either "tasks" (default) to use Ray Tasks or an
                :class:`~ray.data.ActorPoolStrategy` to use an autoscaling actor pool.
        concurrency: The number of Ray workers to use concurrently.

    Returns:
       The `ComputeStrategy` for execution.
    """
    from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
    from ray.data.block import CallableClass
    if isinstance(fn, CallableClass):
        is_callable_class = True
    else:
        is_callable_class = False
        if fn_constructor_args is not None:
            raise ValueError(f'``fn_constructor_args`` can only be specified if providing a callable class instance for ``fn``, but got: {fn}.')
    if compute is not None:
        logger.warning('The argument ``compute`` is deprecated in Ray 2.9. Please specify argument ``concurrency`` instead. For more information, see https://docs.ray.io/en/master/data/transforming-data.html#stateful-transforms.')
        if is_callable_class and (compute == 'tasks' or isinstance(compute, TaskPoolStrategy)):
            raise ValueError(f'``compute`` must specify an actor compute strategy when using a callable class, but got: {compute}. For example, use ``compute=ray.data.ActorPoolStrategy(size=n)``.')
        elif not is_callable_class and (compute == 'actors' or isinstance(compute, ActorPoolStrategy)):
            raise ValueError(f'``compute`` is specified as the actor compute strategy: {compute}, but ``fn`` is not a callable class: {fn}. Pass a callable class or use the default ``compute`` strategy.')
        return compute
    elif concurrency is not None:
        if not is_callable_class:
            logger.warning(f'``concurrency`` is set, but ``fn`` is not a callable class: {fn}. ``concurrency`` are currently only supported when ``fn`` is a callable class.')
            return TaskPoolStrategy()
        if isinstance(concurrency, tuple):
            if len(concurrency) == 2 and isinstance(concurrency[0], int) and isinstance(concurrency[1], int):
                return ActorPoolStrategy(min_size=concurrency[0], max_size=concurrency[1])
            else:
                raise ValueError(f'``concurrency`` is expected to be set as a tuple of integers, but got: {concurrency}.')
        elif isinstance(concurrency, int):
            return ActorPoolStrategy(size=concurrency)
        else:
            raise ValueError(f'``concurrency`` is expected to be set as an integer or a tuple of integers, but got: {concurrency}.')
    elif is_callable_class:
        raise ValueError('``concurrency`` must be specified when using a callable class. For example, use ``concurrency=n`` for a pool of ``n`` workers.')
    else:
        return TaskPoolStrategy()