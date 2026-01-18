import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def get_next_bundle(self) -> RefBundle:
    """Gets the next bundle."""
    assert self.has_bundle()
    if self._min_rows_per_bundle is None:
        assert len(self._bundle_buffer) == 1
        bundle = self._bundle_buffer[0]
        self._bundle_buffer = []
        self._bundle_buffer_size = 0
        return bundle
    leftover = []
    output_buffer = []
    output_buffer_size = 0
    buffer_filled = False
    for bundle in self._bundle_buffer:
        bundle_size = self._get_bundle_size(bundle)
        if buffer_filled:
            leftover.append(bundle)
        elif output_buffer_size + bundle_size <= self._min_rows_per_bundle or output_buffer_size == 0:
            output_buffer.append(bundle)
            output_buffer_size += bundle_size
        else:
            leftover.append(bundle)
            buffer_filled = True
    self._bundle_buffer = leftover
    self._bundle_buffer_size = sum((self._get_bundle_size(bundle) for bundle in leftover))
    return _merge_ref_bundles(*output_buffer)