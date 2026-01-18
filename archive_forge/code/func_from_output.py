import base64
import json
from ray import cloudpickle
from enum import Enum, unique
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import ObjectRef
from ray._private.utils import get_or_create_event_loop
from ray.util.annotations import PublicAPI
@classmethod
def from_output(cls, task_id: str, output: Any):
    """Create static ref from given output."""
    if not isinstance(output, cls):
        if not isinstance(output, ray.ObjectRef):
            output = ray.put(output)
        output = cls(task_id=task_id, ref=output)
    return output