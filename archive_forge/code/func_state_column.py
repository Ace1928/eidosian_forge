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
def state_column(*, filterable: bool, detail: bool=False, format_fn=None, **kwargs):
    """A wrapper around dataclass.field to add additional metadata.

    The metadata is used to define detail / filterable option of
    each column.

    Args:
        detail: If True, the column is used when detail == True
        filterable: If True, the column can be used for filtering.
        kwargs: The same kwargs for the `dataclasses.field` function.
    """
    m = {'detail': detail, 'filterable': filterable, 'format_fn': format_fn}
    if detail and 'default' not in kwargs:
        kwargs['default'] = None
    if 'metadata' in kwargs:
        kwargs['metadata'].update(m)
    else:
        kwargs['metadata'] = m
    return field(**kwargs)