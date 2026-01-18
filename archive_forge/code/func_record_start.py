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
def record_start(self, stats_uuid):
    self.start_time[stats_uuid] = time.perf_counter()
    self.fifo_queue.append(stats_uuid)
    if len(self.fifo_queue) > self.max_stats:
        uuid = self.fifo_queue.pop(0)
        if uuid in self.start_time:
            del self.start_time[uuid]
        if uuid in self.last_time:
            del self.last_time[uuid]
        if uuid in self.metadata:
            del self.metadata[uuid]