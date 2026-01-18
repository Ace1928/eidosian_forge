import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
def dict_to_state(d: Dict, state_schema: StateSchema) -> StateSchema:
    """Convert a dict to a state schema.

    Args:
        d: a dict to convert.
        state_schema: a schema to convert to.

    Returns:
        A state schema.
    """
    try:
        return resource_to_schema(state_schema)(**d)
    except Exception as e:
        raise RayStateApiException(f'Failed to convert {d} to StateSchema: {e}') from e