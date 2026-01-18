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
def map_block_split(self, input_files: List[str], num_blocks: int, *blocks_and_fn_args, **fn_kwargs) -> BlockPartition:
    return _map_block_split(block_fn, input_files, self.fn, num_blocks, *blocks_and_fn_args, **fn_kwargs)