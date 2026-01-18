import logging
import os
import subprocess
from typing import Optional
from wandb.sdk.launch.errors import LaunchError
from wandb.util import get_module
from ..utils import GCS_URI_RE, event_loop_thread_exec
from .abstract import AbstractEnvironment
def get_default_region() -> Optional[str]:
    """Get the default region from gcloud config or environment variables.

    Returns:
        str: The default region, or None if it cannot be determined.
    """
    region = get_gcloud_config_value('compute/region')
    if not region:
        region = os.environ.get(GCP_REGION_ENV_VAR)
    return region