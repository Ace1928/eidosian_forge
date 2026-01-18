import asyncio
import logging
import queue
from typing import Dict, List, Set, Optional, TYPE_CHECKING
import ray
from ray.workflow import common
from ray.workflow.common import WorkflowStatus, TaskID
from ray.workflow import workflow_state_from_storage
from ray.workflow import workflow_context
from ray.workflow import workflow_storage
from ray.workflow.exceptions import (
from ray.workflow.workflow_executor import WorkflowExecutor
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.workflow_context import WorkflowTaskContext
def gen_task_id(self, workflow_id: str, task_name: str) -> str:
    wf_store = workflow_storage.WorkflowStorage(workflow_id)
    idx = wf_store.gen_task_id(task_name)
    if idx == 0:
        return task_name
    else:
        return f'{task_name}_{idx}'