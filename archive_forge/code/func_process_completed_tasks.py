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
def process_completed_tasks(topology: Topology, backpressure_policies: List[BackpressurePolicy], max_errored_blocks: int) -> int:
    """Process any newly completed tasks. To update operator
    states, call `update_operator_states()` afterwards.

    Args:
        topology: The toplogy of operators.
        backpressure_policies: The backpressure policies to use.
        max_errored_blocks: Max number of errored blocks to allow,
            unlimited if negative.
    Returns:
        The number of errored blocks.
    """
    active_tasks: Dict[Waitable, Tuple[OpState, OpTask]] = {}
    for op, state in topology.items():
        for task in op.get_active_tasks():
            active_tasks[task.get_waitable()] = (state, task)
    max_blocks_to_read_per_op: Dict[OpState, int] = {}
    for policy in backpressure_policies:
        res = policy.calculate_max_blocks_to_read_per_op(topology)
        if len(res) > 0:
            if len(max_blocks_to_read_per_op) > 0:
                raise ValueError('At most one backpressure policy that implements calculate_max_blocks_to_read_per_op() can be used at a time.')
            else:
                max_blocks_to_read_per_op = res
    num_errored_blocks = 0
    if active_tasks:
        ready, _ = ray.wait(list(active_tasks.keys()), num_returns=len(active_tasks), fetch_local=False, timeout=0.1)
        ready_tasks_by_op = defaultdict(list)
        for ref in ready:
            state, task = active_tasks[ref]
            ready_tasks_by_op[state].append(task)
        for state, ready_tasks in ready_tasks_by_op.items():
            ready_tasks = sorted(ready_tasks, key=lambda t: t.task_index())
            for task in ready_tasks:
                if isinstance(task, DataOpTask):
                    try:
                        num_blocks_read = task.on_data_ready(max_blocks_to_read_per_op.get(state, None))
                        if state in max_blocks_to_read_per_op:
                            max_blocks_to_read_per_op[state] -= num_blocks_read
                    except Exception as e:
                        num_errored_blocks += 1
                        should_ignore = max_errored_blocks < 0 or max_errored_blocks >= num_errored_blocks
                        error_message = f'An exception was raised from a task of operator "{state.op.name}".'
                        if should_ignore:
                            remaining = max_errored_blocks - num_errored_blocks if max_errored_blocks >= 0 else 'unlimited'
                            error_message += f' Ignoring this exception with remaining max_errored_blocks={remaining}.'
                            logger.get_logger().warning(error_message, exc_info=e)
                        else:
                            error_message += ' Dataset execution will now abort. To ignore this exception and continue, set DataContext.max_errored_blocks.'
                            logger.get_logger().error(error_message)
                            raise e from None
                else:
                    assert isinstance(task, MetadataOpTask)
                    task.on_task_finished()
    for op, op_state in topology.items():
        while op.has_next():
            op_state.add_output(op.get_next())
    return num_errored_blocks