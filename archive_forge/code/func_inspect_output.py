import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
def inspect_output(self, task_id: TaskID) -> Optional[TaskID]:
    """Get the actual checkpointed output for a task, represented by the ID of
        the task that actually keeps the checkpoint.

        Raises:
            ValueError: The workflow does not exist or the workflow state is not valid.

        Args:
            task_id: The ID of the task we are looking for its checkpoint.

        Returns:
            The ID of the task that actually keeps the checkpoint.
                'None' if the checkpoint does not exist.
        """
    status = self.load_workflow_status()
    if status == WorkflowStatus.NONE:
        raise ValueError(f"No such workflow '{self._workflow_id}'")
    if status == WorkflowStatus.CANCELED:
        raise ValueError(f'Workflow {self._workflow_id} is canceled')
    if status == WorkflowStatus.RESUMABLE:
        raise ValueError(f'Workflow {self._workflow_id} is in resumable status, please resume it')
    if task_id is None:
        task_id = self.get_entrypoint_task_id()
    return self._locate_output_in_storage(task_id)