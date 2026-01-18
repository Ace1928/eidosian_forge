from typing import Dict, List, Iterator, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import time
from collections import defaultdict
import ray
from ray.exceptions import RayTaskError, RayError
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowCancellationError, WorkflowExecutionError
from ray.workflow.task_executor import get_task_executor, _BakedWorkflowInputs
from ray.workflow.workflow_state import (
def get_task_output_async(self, task_id: Optional[TaskID]) -> asyncio.Future:
    """Get the output of a task asynchronously.

        Args:
            task_id: The ID of task the callback associates with.

        Returns:
            A callback in the form of a future that associates with the task.
        """
    state = self._state
    if self._task_done_callbacks[task_id]:
        return self._task_done_callbacks[task_id][0]
    fut = asyncio.Future()
    task_id = state.continuation_root.get(task_id, task_id)
    output = state.get_input(task_id)
    if output is not None:
        fut.set_result(output)
    elif task_id in state.done_tasks:
        fut.set_exception(ValueError(f"Task '{task_id}' is done but neither in memory or in storage could we find its output. It could because its in memory output has been garbage collected and the task did notcheckpoint its output."))
    else:
        self._task_done_callbacks[task_id].append(fut)
    return fut