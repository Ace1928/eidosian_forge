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
def select_operator_to_run(topology: Topology, cur_usage: TopologyResourceUsage, limits: ExecutionResources, backpressure_policies: List[BackpressurePolicy], ensure_at_least_one_running: bool, execution_id: str, autoscaling_state: AutoscalingState) -> Optional[PhysicalOperator]:
    """Select an operator to run, if possible.

    The objective of this function is to maximize the throughput of the overall
    pipeline, subject to defined memory and parallelism limits.

    This is currently implemented by applying backpressure on operators that are
    producing outputs faster than they are consuming them `len(outqueue)`, as well as
    operators with a large number of running tasks `num_processing()`.

    Note that memory limits also apply to the outqueue of the output operator. This
    provides backpressure if the consumer is slow. However, once a bundle is returned
    to the user, it is no longer tracked.
    """
    assert isinstance(cur_usage, TopologyResourceUsage), cur_usage
    ops = []
    for op, state in topology.items():
        under_resource_limits = _execution_allowed(op, cur_usage, limits)
        if op.need_more_inputs() and state.num_queued() > 0 and op.should_add_input() and under_resource_limits and (not op.completed()) and all((p.can_add_input(op) for p in backpressure_policies)):
            ops.append(op)
        op.notify_resource_usage(state.num_queued(), under_resource_limits)
    if not ops and any((state.num_queued() > 0 for state in topology.values())):
        now = time.time()
        if now > autoscaling_state.last_request_ts + MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS:
            autoscaling_state.last_request_ts = now
            _try_to_scale_up_cluster(topology, execution_id)
    if ensure_at_least_one_running and (not ops) and all((op.num_active_tasks() == 0 for op in topology)):
        ops = [op for op, state in topology.items() if op.need_more_inputs() and state.num_queued() > 0 and (not op.completed())]
    if not ops:
        return None
    return min(ops, key=lambda op: (not op.throttling_disabled(), len(topology[op].outqueue) + topology[op].num_processing()))