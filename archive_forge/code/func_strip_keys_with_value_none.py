import dataclasses
import logging
import os
import re
import traceback
from dataclasses import dataclass
from typing import Iterator, List, Optional, Any, Dict, Tuple, Union
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.pydantic_models import (
from ray.dashboard.modules.job.common import (
from ray.runtime_env import RuntimeEnv
def strip_keys_with_value_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Strip keys with value None from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}