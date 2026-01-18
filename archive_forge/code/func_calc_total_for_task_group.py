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
def calc_total_for_task_group(task_group: NestedTaskSummary) -> NestedTaskSummary:
    """
            Calculates the total of a group as the sum of all children.
            Sorts children by timestamp
            """
    if not len(task_group.children):
        return task_group
    for child in task_group.children:
        totaled = calc_total_for_task_group(child)
        for state, count in totaled.state_counts.items():
            task_group.state_counts[state] = task_group.state_counts.get(state, 0) + count
    sort_task_groups(task_group.children)
    return task_group