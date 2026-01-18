from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@property
def average_bytes_outputs_per_task(self) -> Optional[float]:
    """Average size in bytes of output blocks per task,
        or None if no task has finished."""
    if self.num_tasks_finished == 0:
        return None
    else:
        return self.bytes_outputs_of_finished_tasks / self.num_tasks_finished