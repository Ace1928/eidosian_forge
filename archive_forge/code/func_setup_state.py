import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
def setup_state(op: PhysicalOperator) -> OpState:
    if op in topology:
        raise ValueError('An operator can only be present in a topology once.')
    inqueues = []
    for i, parent in enumerate(op.input_dependencies):
        parent_state = setup_state(parent)
        inqueues.append(parent_state.outqueue)
    op_state = OpState(op, inqueues)
    topology[op] = op_state
    op.start(options)
    return op_state