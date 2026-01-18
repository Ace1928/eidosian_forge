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
def get_workflow_storage(workflow_id: Optional[str]=None) -> WorkflowStorage:
    """Get the storage for the workflow.

    Args:
        workflow_id: The ID of the storage.

    Returns:
        A workflow storage.
    """
    if workflow_id is None:
        workflow_id = workflow_context.get_workflow_task_context().workflow_id
    return WorkflowStorage(workflow_id)