from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
@root_validator(pre=True)
@classmethod
def check_extra_fields(cls, values: dict) -> dict:
    """Check for extra fields and print a warning."""
    for key in values:
        if key not in ['type', 'region']:
            wandb.termwarn(f'Unrecognized field {key} in environment block. Please check your config file.')
    return values