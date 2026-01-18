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
@ray.remote(num_cpus=0)
def resume_workflow_task(job_id: str, workflow_id: str, task_id: Optional[TaskID]=None) -> WorkflowExecutionState:
    """Resume a task of a workflow.

    Args:
        job_id: The ID of the job that submits the workflow execution. The ID
        is used to identify the submitter of the workflow.
        workflow_id: The ID of the workflow job. The ID is used to identify
            the workflow.
        task_id: The task to resume in the workflow.

    Raises:
        WorkflowNotResumableException: fail to resume the workflow.

    Returns:
        The execution result of the workflow, represented by Ray ObjectRef.
    """
    with workflow_context.workflow_logging_context(job_id):
        try:
            return workflow_state_from_storage.workflow_state_from_storage(workflow_id, task_id)
        except Exception as e:
            raise WorkflowNotResumableError(workflow_id) from e