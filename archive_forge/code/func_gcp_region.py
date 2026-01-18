import asyncio
import logging
from typing import Any, Dict, Optional
from wandb.apis.internal import Api
from wandb.util import get_module
from .._project_spec import LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..environment.gcp_environment import GcpEnvironment
from ..errors import LaunchError
from ..registry.abstract import AbstractRegistry
from ..utils import MAX_ENV_LENGTHS, PROJECT_SYNCHRONOUS, event_loop_thread_exec
from .abstract import AbstractRun, AbstractRunner, Status
@property
def gcp_region(self) -> str:
    return self._job.location