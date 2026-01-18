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
def get_entrypoint_task_id(self) -> TaskID:
    """Load the entrypoint task ID of the workflow.

        Returns:
            The ID of the entrypoint task.
        """
    try:
        return self._locate_output_task_id('')
    except Exception as e:
        raise ValueError(f'Fail to get entrypoint task ID from workflow[id={self._workflow_id}]') from e