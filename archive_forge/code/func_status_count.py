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
@classmethod
def status_count(cls) -> Dict[State, int]:
    """Get a dictionary mapping statuses to the # monitored jobs with each status."""
    if cls._instance is None:
        raise ValueError('LaunchKubernetesMonitor not initialized, cannot get status counts.')
    return cls._instance.__status_count()