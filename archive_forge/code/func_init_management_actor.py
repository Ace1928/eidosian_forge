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
def init_management_actor(max_running_workflows: Optional[int], max_pending_workflows: Optional[int]) -> None:
    """Initialize WorkflowManagementActor.

    Args:
        max_running_workflows: The maximum number of concurrently running workflows.
            Use -1 as infinity. Use 'None' for keeping the original value if the actor
            exists, or it is equivalent to infinity if the actor does not exist.
        max_pending_workflows: The maximum number of queued workflows.
            Use -1 as infinity. Use 'None' for keeping the original value if the actor
            exists, or it is equivalent to infinity if the actor does not exist.
    """
    try:
        actor = get_management_actor()
        ray.get(actor.validate_init_options.remote(max_running_workflows, max_pending_workflows))
    except ValueError:
        logger.info('Initializing workflow manager...')
        if max_running_workflows is None:
            max_running_workflows = -1
        if max_pending_workflows is None:
            max_pending_workflows = -1
        actor = WorkflowManagementActor.options(name=common.MANAGEMENT_ACTOR_NAME, namespace=common.MANAGEMENT_ACTOR_NAMESPACE, lifetime='detached').remote(max_running_workflows, max_pending_workflows)
        ray.get(actor.ready.remote())