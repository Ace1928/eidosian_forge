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
def get_output_blocking(self, output_split_idx: Optional[int]) -> RefBundle:
    """Get an item from this node's output queue, blocking as needed.

        Returns:
            The RefBundle from the output queue, or an error / end of stream indicator.

        Raises:
            StopIteration: If all outputs are already consumed.
            Exception: If there was an exception raised during execution.
        """
    while True:
        if self._exception is not None:
            raise self._exception
        elif self._finished and (not self.outqueue.has_next(output_split_idx)):
            raise StopIteration()
        ref = self.outqueue.pop(output_split_idx)
        if ref is not None:
            return ref
        time.sleep(0.01)