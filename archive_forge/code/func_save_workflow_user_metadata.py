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
def save_workflow_user_metadata(self, metadata: Dict[str, Any]):
    """Save user metadata of the current workflow.

        Args:
            metadata: user metadata of the current workflow.

        Raises:
            DataSaveError: if we fail to save the user metadata.
        """
    self._put(self._key_workflow_user_metadata(), metadata, True)