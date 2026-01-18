import functools
import logging
from typing import Dict, Set, List, Tuple, Union, Optional, Any
import time
import uuid
import ray
from ray.dag import DAGNode
from ray.dag.input_node import DAGInputData
from ray.remote_function import RemoteFunction
from ray.workflow.common import (
from ray.workflow import serialization, workflow_access, workflow_context
from ray.workflow.event_listener import EventListener, EventListenerType, TimerListener
from ray.workflow.workflow_storage import WorkflowStorage
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
from ray.util.annotations import PublicAPI
from ray._private.usage import usage_lib
@PublicAPI(stability='alpha')
def resume_async(workflow_id: str) -> ray.ObjectRef:
    """Resume a workflow asynchronously.

    Resume a workflow and retrieve its output. If the workflow was incomplete,
    it will be re-executed from its checkpointed outputs. If the workflow was
    complete, returns the result immediately.

    Examples:
        .. testcode::

            from ray import workflow

            @ray.remote
            def start_trip():
                return 3

            trip = start_trip.bind()
            res1 = workflow.run_async(trip, workflow_id="trip1")
            res2 = workflow.resume_async("trip1")
            assert ray.get(res1) == ray.get(res2)

    Args:
        workflow_id: The id of the workflow to resume.

    Returns:
        An object reference that can be used to retrieve the workflow result.
    """
    _ensure_workflow_initialized()
    logger.info(f'Resuming workflow [id="{workflow_id}"].')
    workflow_manager = workflow_access.get_management_actor()
    if ray.get(workflow_manager.is_workflow_non_terminating.remote(workflow_id)):
        raise RuntimeError(f"Workflow '{workflow_id}' is already running or pending.")
    job_id = ray.get_runtime_context().get_job_id()
    context = workflow_context.WorkflowTaskContext(workflow_id=workflow_id)
    ray.get(workflow_manager.reconstruct_workflow.remote(job_id, context))
    result = workflow_manager.execute_workflow.remote(job_id, context)
    logger.info(f'Workflow job {workflow_id} resumed.')
    return result