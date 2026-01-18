from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
def on_task_finished(self, task_index: int, exception: Optional[Exception]):
    """Callback when a task is finished."""
    self.num_tasks_running -= 1
    self.num_tasks_finished += 1
    if exception is not None:
        self.num_tasks_failed += 1
    task_info = self._running_tasks[task_index]
    self.num_outputs_of_finished_tasks += task_info.num_outputs
    self.bytes_outputs_of_finished_tasks += task_info.bytes_outputs
    inputs = self._running_tasks[task_index].inputs
    self.num_inputs_processed += len(inputs)
    total_input_size = inputs.size_bytes()
    self.bytes_inputs_processed += total_input_size
    blocks = [input[0] for input in inputs.blocks]
    metadata = [input[1] for input in inputs.blocks]
    ctx = ray.data.context.DataContext.get_current()
    if ctx.enable_get_object_locations_for_metrics:
        locations = ray.experimental.get_object_locations(blocks)
        for block, meta in zip(blocks, metadata):
            if locations[block].get('did_spill', False):
                assert meta.size_bytes is not None
                self.obj_store_mem_spilled += meta.size_bytes
    self.obj_store_mem_freed += total_input_size
    self.obj_store_mem_cur -= total_input_size
    inputs.destroy_if_owned()
    del self._running_tasks[task_index]