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
def print_status(self) -> None:
    """Prints the current status of the agent."""
    output_str = 'agent '
    if self._name:
        output_str += f'{self._name} '
    if self.num_running_jobs < self._max_jobs:
        output_str += 'polling on '
        if self._project != LAUNCH_DEFAULT_PROJECT:
            output_str += f'project {self._project}, '
        output_str += f'queues {','.join(self._queues)}, '
    output_str += f'running {self.num_running_jobs} out of a maximum of {self._max_jobs} jobs'
    wandb.termlog(f'{LOG_PREFIX}{output_str}')
    if self.num_running_jobs > 0:
        output_str += f': {','.join((str(job_id) for job_id in self.thread_ids))}'
    _logger.info(output_str)