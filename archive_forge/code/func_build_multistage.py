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
def build_multistage(self, stages: StatsDict) -> 'DatasetStats':
    stage_infos = {}
    for i, (k, v) in enumerate(stages.items()):
        capped_k = capfirst(k)
        if len(stages) > 1:
            if i == 0:
                stage_infos[self.stage_name + capped_k] = v
            else:
                stage_infos[self.stage_name.split('->')[-1] + capped_k] = v
        else:
            stage_infos[self.stage_name] = v
    stats = DatasetStats(stages=stage_infos, parent=self.parent, base_name=self.stage_name)
    stats.time_total_s = time.perf_counter() - self.start_time
    return stats