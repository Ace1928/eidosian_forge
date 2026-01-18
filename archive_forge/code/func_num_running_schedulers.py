import asyncio
import logging
import os
import pprint
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Event
from typing import Any, Dict, List, Optional, Union
import wandb
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.runner.local_container import LocalSubmittedRun
from wandb.sdk.launch.runner.local_process import LocalProcessRunner
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import runid
from .. import loader
from .._project_spec import LaunchProject
from ..builder.build import construct_agent_configs
from ..errors import LaunchDockerError, LaunchError
from ..utils import (
from .job_status_tracker import JobAndRunStatusTracker
from .run_queue_item_file_saver import RunQueueItemFileSaver
@property
def num_running_schedulers(self) -> int:
    """Return just the number of schedulers."""
    with self._jobs_lock:
        return len([x for x in self._jobs if self._jobs[x].is_scheduler])