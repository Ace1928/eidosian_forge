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
def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
    """Get the status of the workflow."""
    if workflow_id in self._workflow_executors:
        if workflow_id in self._queued_workflows:
            return WorkflowStatus.PENDING
        return WorkflowStatus.RUNNING
    store = workflow_storage.get_workflow_storage(workflow_id)
    status = store.load_workflow_status()
    if status == WorkflowStatus.NONE:
        raise WorkflowNotFoundError(workflow_id)
    elif status in WorkflowStatus.non_terminating_status():
        return WorkflowStatus.RESUMABLE
    return status