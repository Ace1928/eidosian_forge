from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
def on_output_taken(self, output: RefBundle):
    """Callback when an output is taken from the operator."""
    output_bytes = output.size_bytes()
    self.num_outputs_taken += 1
    self.bytes_outputs_taken += output_bytes
    self.obj_store_mem_cur -= output_bytes