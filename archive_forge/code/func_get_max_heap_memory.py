import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def get_max_heap_memory(self) -> float:
    parent_memory = [p.get_max_heap_memory() for p in self.parents]
    parent_max = max(parent_memory) if parent_memory else 0
    if not self.stages_stats:
        return parent_max
    return max(parent_max, *[ss.memory.get('max', 0) for ss in self.stages_stats])