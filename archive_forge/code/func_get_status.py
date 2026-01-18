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
def get_status(workflow_id: str) -> WorkflowStatus:
    """Get the status for a given workflow.

    Args:
        workflow_id: The workflow to query.

    Examples:
        .. testcode::

            from ray import workflow

            @ray.remote
            def trip():
                pass

            workflow_task = trip.bind()
            output = workflow.run(workflow_task, workflow_id="local_trip")
            assert workflow.SUCCESSFUL == workflow.get_status("local_trip")

    Returns:
        The status of that workflow
    """
    _ensure_workflow_initialized()
    if not isinstance(workflow_id, str):
        raise TypeError('workflow_id has to be a string type.')
    workflow_manager = workflow_access.get_management_actor()
    return ray.get(workflow_manager.get_workflow_status.remote(workflow_id))