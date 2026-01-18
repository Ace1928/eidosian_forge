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
def run_async(dag: DAGNode, *args, workflow_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs) -> ray.ObjectRef:
    """Run a workflow asynchronously.

    If the workflow with the given id already exists, it will be resumed.

    Args:
        workflow_id: A unique identifier that can be used to resume the
            workflow. If not specified, a random id will be generated.
        metadata: The metadata to add to the workflow. It has to be able
            to serialize to json.

    Returns:
       The running result as ray.ObjectRef.

    """
    _ensure_workflow_initialized()
    if not isinstance(dag, DAGNode):
        raise TypeError('Input should be a DAG.')
    input_data = DAGInputData(*args, **kwargs)
    validate_user_metadata(metadata)
    metadata = metadata or {}
    if workflow_id is None:
        workflow_id = f'{str(uuid.uuid4())}.{time.time():.9f}'
    workflow_manager = workflow_access.get_management_actor()
    if ray.get(workflow_manager.is_workflow_non_terminating.remote(workflow_id)):
        raise RuntimeError(f"Workflow '{workflow_id}' is already running or pending.")
    state = workflow_state_from_dag(dag, input_data, workflow_id)
    logger.info(f'Workflow job created. [id="{workflow_id}"].')
    context = workflow_context.WorkflowTaskContext(workflow_id=workflow_id)
    with workflow_context.workflow_task_context(context):

        @client_mode_wrap
        def _try_checkpoint_workflow(workflow_state) -> bool:
            ws = WorkflowStorage(workflow_id)
            ws.save_workflow_user_metadata(metadata)
            try:
                ws.get_entrypoint_task_id()
                return True
            except Exception:
                ws.save_workflow_execution_state('', workflow_state)
                return False
        wf_exists = _try_checkpoint_workflow(state)
        if wf_exists:
            return resume_async(workflow_id)
        ray.get(workflow_manager.submit_workflow.remote(workflow_id, state, ignore_existing=False))
        job_id = ray.get_runtime_context().get_job_id()
        return workflow_manager.execute_workflow.remote(job_id, context)