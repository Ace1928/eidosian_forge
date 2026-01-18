from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@property
def average_bytes_change_per_task(self) -> Optional[float]:
    """Average size difference in bytes of input ref bundles and output ref
        bundles per task."""
    if self.average_bytes_inputs_per_task is None or self.average_bytes_outputs_per_task is None:
        return None
    return self.average_bytes_outputs_per_task - self.average_bytes_inputs_per_task