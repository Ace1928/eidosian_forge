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
def create_http_event_provider(self) -> None:
    """Deploy an HTTPEventProvider as a Serve deployment with
        name = common.HTTP_EVENT_PROVIDER_NAME, if one doesn't exist
        """
    ray.serve.start(detached=True)
    provider_exists = common.HTTP_EVENT_PROVIDER_NAME in ray.serve.status().applications
    if not provider_exists:
        from ray.workflow.http_event_provider import HTTPEventProvider
        ray.serve.run(HTTPEventProvider.bind(), name=common.HTTP_EVENT_PROVIDER_NAME, route_prefix='/event')