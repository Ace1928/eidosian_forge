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
@client_mode_wrap
def resume_all(include_failed: bool=False) -> List[Tuple[str, ray.ObjectRef]]:
    """Resume all resumable workflow jobs.

    This can be used after cluster restart to resume all tasks.

    Args:
        include_failed: Whether to resume FAILED workflows.

    Examples:
        .. testcode::

            from ray import workflow

            @ray.remote
            def failed_job():
                raise ValueError()

            workflow_task = failed_job.bind()
            output = workflow.run_async(
                workflow_task, workflow_id="failed_job")
            try:
                ray.get(output)
            except Exception:
                print("JobFailed")

            assert workflow.get_status("failed_job") == workflow.FAILED
            print(workflow.resume_all(include_failed=True))

        .. testoutput::

            JobFailed
            [('failed_job', ObjectRef(...))]

    Returns:
        A list of (workflow_id, returned_obj_ref) resumed.
    """
    _ensure_workflow_initialized()
    filter_set = {WorkflowStatus.RESUMABLE}
    if include_failed:
        filter_set.add(WorkflowStatus.FAILED)
    all_failed = list_all(filter_set)
    try:
        workflow_manager = workflow_access.get_management_actor()
    except Exception as e:
        raise RuntimeError('Failed to get management actor') from e
    job_id = ray.get_runtime_context().get_job_id()
    reconstructed_workflows = []
    for wid, _ in all_failed:
        context = workflow_context.WorkflowTaskContext(workflow_id=wid)
        try:
            ray.get(workflow_manager.reconstruct_workflow.remote(job_id, context))
        except Exception as e:
            logger.error(f'Failed to resume workflow {context.workflow_id}', exc_info=e)
            raise
        reconstructed_workflows.append(context)
    results = []
    for context in reconstructed_workflows:
        results.append((context.workflow_id, workflow_manager.execute_workflow.remote(job_id, context)))
    return results