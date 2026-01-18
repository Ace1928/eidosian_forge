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
def merge_sibings_for_task_group(siblings: List[NestedTaskSummary]) -> Tuple[List[NestedTaskSummary], Optional[int]]:
    """
            Merges task summaries with the same name into a group if there are more than
            one child with that name.

            Args:
                siblings: A list of NestedTaskSummary's to merge together

            Returns
                Index 0: A list of NestedTaskSummary's which have been merged
                Index 1: The smallest timestamp amongst the siblings
            """
    if not len(siblings):
        return (siblings, None)
    groups = {}
    min_timestamp = None
    for child in siblings:
        child.children, child_min_timestamp = merge_sibings_for_task_group(child.children)
        if child_min_timestamp and child_min_timestamp < (child.timestamp or sys.maxsize):
            child.timestamp = child_min_timestamp
        if child.name not in groups:
            groups[child.name] = NestedTaskSummary(name=child.name, key=child.name, type='GROUP')
        groups[child.name].children.append(child)
        if child.timestamp and child.timestamp < (groups[child.name].timestamp or sys.maxsize):
            groups[child.name].timestamp = child.timestamp
            if child.timestamp < (min_timestamp or sys.maxsize):
                min_timestamp = child.timestamp
    return ([group if len(group.children) > 1 else group.children[0] for group in groups.values()], min_timestamp)