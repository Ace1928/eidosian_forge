from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@property
def average_bytes_inputs_per_task(self) -> Optional[float]:
    """Average size in bytes of ref bundles passed to tasks, or ``None`` if no
        tasks have been submitted."""
    if self.num_tasks_submitted == 0:
        return None
    else:
        return self.bytes_inputs_of_submitted_tasks / self.num_tasks_submitted