import logging
import os
import subprocess
from typing import Optional
from wandb.sdk.launch.errors import LaunchError
from wandb.util import get_module
from ..utils import GCS_URI_RE, event_loop_thread_exec
from .abstract import AbstractEnvironment
def get_gcloud_config_value(config_name: str) -> Optional[str]:
    """Get a value from gcloud config.

    Arguments:
        config_name: The name of the config value.

    Returns:
        str: The config value, or None if the value is not set.
    """
    try:
        output = subprocess.check_output(['gcloud', 'config', 'get-value', config_name], stderr=subprocess.STDOUT)
        value = str(output.decode('utf-8').strip())
        if value and 'unset' not in value:
            return value
        return None
    except subprocess.CalledProcessError:
        return None