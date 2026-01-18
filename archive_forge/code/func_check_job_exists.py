import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def check_job_exists(public_api: 'PublicApi', job: Optional[str]) -> bool:
    """Check if the job exists using the public api.

    Returns: True if no job is passed, or if the job exists.
    Returns: False if the job is misformatted or doesn't exist.
    """
    if not job:
        return True
    try:
        public_api.job(job)
    except Exception as e:
        wandb.termerror(f'Failed to load job. {e}')
        return False
    return True