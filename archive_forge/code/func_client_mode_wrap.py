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
def client_mode_wrap(func):
    """Wraps a function called during client mode for execution as a remote task.

    Adopted from "ray._private.client_mode_hook.client_mode_wrap". Some changes are made
    (e.g., init the workflow instead of init Ray; the latter does not specify a storage
    during Ray init and will result in workflow failures).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from ray._private.client_mode_hook import client_mode_should_convert
        from ray._private.auto_init_hook import enable_auto_connect
        if enable_auto_connect:
            _ensure_workflow_initialized()
        if client_mode_should_convert():
            f = ray.remote(num_cpus=0)(func)
            ref = f.remote(*args, **kwargs)
            return ray.get(ref)
        return func(*args, **kwargs)
    return wrapper