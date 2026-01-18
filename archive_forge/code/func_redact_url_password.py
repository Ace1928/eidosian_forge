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
def redact_url_password(url: str) -> str:
    """Redact any passwords in a URL."""
    secret = re.findall('https?:\\/\\/.*:(.*)@.*', url)
    if len(secret) > 0:
        url = url.replace(f':{secret[0]}@', ':<redacted>@')
    return url