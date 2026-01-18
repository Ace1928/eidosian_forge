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
def save_actor_class_body(self, cls: type) -> None:
    """Save the class body of the virtual actor.

        Args:
            cls: The class body used by the virtual actor.

        Raises:
            DataSaveError: if we fail to save the class body.
        """
    self._put(self._key_class_body(), cls)