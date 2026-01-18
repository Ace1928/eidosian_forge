import collections
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
def get_compute(compute_spec: Union[str, ComputeStrategy]) -> ComputeStrategy:
    if not isinstance(compute_spec, (TaskPoolStrategy, ActorPoolStrategy)):
        raise ValueError(f'In Ray 2.5, the compute spec must be either TaskPoolStrategy or ActorPoolStategy, was: {compute_spec}.')
    elif not compute_spec or compute_spec == 'tasks':
        return TaskPoolStrategy()
    elif compute_spec == 'actors':
        return ActorPoolStrategy()
    elif isinstance(compute_spec, ComputeStrategy):
        return compute_spec
    else:
        raise ValueError('compute must be one of [`tasks`, `actors`, ComputeStrategy]')