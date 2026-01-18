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
def monitor_namespace(cls, namespace: str, custom_resource: Optional[CustomResource]=None) -> None:
    """Start monitoring a namespaces for resources."""
    if cls._instance is None:
        raise LaunchError('LaunchKubernetesMonitor not initialized, cannot monitor namespace.')
    cls._instance.__monitor_namespace(namespace, custom_resource=custom_resource)