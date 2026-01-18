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
def dispatch_next_task(self) -> None:
    """Move a bundle from the operator inqueue to the operator itself."""
    for i, inqueue in enumerate(self.inqueues):
        ref = inqueue.pop()
        if ref is not None:
            self.op.add_input(ref, input_index=i)
            return
    assert False, 'Nothing to dispatch'