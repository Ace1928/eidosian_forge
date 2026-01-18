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
def inspect_task(self, task_id: TaskID) -> TaskInspectResult:
    """
        Get the status of a workflow task. The status indicates whether
        the workflow task can be recovered etc.

        Args:
            task_id: The ID of a workflow task

        Returns:
            The status of the task.
        """
    return self._inspect_task(task_id)