import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import kubernetes_asyncio  # type: ignore # noqa: F401
import urllib3
from kubernetes_asyncio import watch
from kubernetes_asyncio.client import (  # type: ignore  # noqa: F401
import wandb
from wandb.sdk.launch.agent import LaunchAgent
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.runner.abstract import State, Status
from wandb.sdk.launch.utils import get_kube_context_and_api_client
def create_named_task(name: str, coro: Any, *args: Any, **kwargs: Any) -> asyncio.Task:
    """Create a named task."""
    task = asyncio.create_task(coro(*args, **kwargs))
    if sys.version_info >= (3, 8):
        task.set_name(name)
    task.add_done_callback(_log_err_task_callback)
    return task