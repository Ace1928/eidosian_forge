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
def delete_workflow(self, workflow_id: str) -> None:
    """Delete a workflow, its checkpoints, and other information it may have
           persisted to storage.

        Args:
            workflow_id: The workflow to delete.

        Raises:
            WorkflowStillActiveError: The workflow is still active.
            WorkflowNotFoundError: The workflow does not exist.
        """
    if self.is_workflow_non_terminating(workflow_id):
        raise WorkflowStillActiveError('DELETE', workflow_id)
    wf_storage = workflow_storage.WorkflowStorage(workflow_id)
    wf_storage.delete_workflow()
    self._executed_workflows.discard(workflow_id)