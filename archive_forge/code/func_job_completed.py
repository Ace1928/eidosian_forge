import logging
from dataclasses import dataclass
from typing import Optional
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch._project_spec import LaunchProject
from ..runner.abstract import AbstractRun
from ..utils import event_loop_thread_exec
from .run_queue_item_file_saver import RunQueueItemFileSaver
@property
def job_completed(self) -> bool:
    return self.failed_to_start or self.completed_status is not None