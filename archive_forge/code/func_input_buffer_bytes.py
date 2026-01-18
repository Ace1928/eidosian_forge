from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@property
def input_buffer_bytes(self) -> int:
    """Size in bytes of input blocks that are not processed yet."""
    return self.bytes_inputs_received - self.bytes_inputs_processed