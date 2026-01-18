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
def sort_task_groups(task_groups: List[NestedTaskSummary]) -> None:
    task_groups.sort(key=lambda x: 0 if x.type == 'ACTOR_CREATION_TASK' else 1)
    task_groups.sort(key=lambda x: x.timestamp or sys.maxsize)
    task_groups.sort(key=lambda x: x.state_counts.get('FAIELD', 0), reverse=True)
    task_groups.sort(key=get_pending_tasks_count, reverse=True)
    task_groups.sort(key=get_running_tasks_count, reverse=True)