from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
def on_input_received(self, input: RefBundle):
    """Callback when the operator receives a new input."""
    self.num_inputs_received += 1
    input_size = input.size_bytes()
    self.bytes_inputs_received += input_size
    self.obj_store_mem_cur += input_size
    if self.obj_store_mem_cur > self.obj_store_mem_peak:
        self.obj_store_mem_peak = self.obj_store_mem_cur