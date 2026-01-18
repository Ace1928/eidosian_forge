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
def continuation(dag_node: 'DAGNode') -> Union['DAGNode', Any]:
    """Converts a DAG into a continuation.

    The result depends on the context. If it is inside a workflow, it
    returns a workflow; otherwise it executes and get the result of
    the DAG.

    Args:
        dag_node: The DAG to be converted.
    """
    from ray.workflow.workflow_context import in_workflow_execution
    if not isinstance(dag_node, DAGNode):
        raise TypeError('Input should be a DAG.')
    if in_workflow_execution():
        return dag_node
    return ray.get(dag_node.execute())