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
def build_streaming_topology(dag: PhysicalOperator, options: ExecutionOptions) -> Tuple[Topology, int]:
    """Instantiate the streaming operator state topology for the given DAG.

    This involves creating the operator state for each operator in the DAG,
    registering it with this class, and wiring up the inqueues/outqueues of
    dependent operator states.

    Args:
        dag: The operator DAG to instantiate.
        options: The execution options to use to start operators.

    Returns:
        The topology dict holding the streaming execution state.
        The number of progress bars initialized so far.
    """
    topology: Topology = {}

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
    setup_state(dag)
    i = 1
    for op_state in list(topology.values()):
        if not isinstance(op_state.op, InputDataBuffer):
            i += op_state.initialize_progress_bars(i, options.verbose_progress)
    return (topology, i)