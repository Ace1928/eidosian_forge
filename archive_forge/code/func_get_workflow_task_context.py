import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import ray
from ray._private.ray_logging import configure_log_file, get_worker_log_file_name
from ray.workflow.common import CheckpointModeType, WorkflowStatus
def get_workflow_task_context() -> Optional[WorkflowTaskContext]:
    return _context