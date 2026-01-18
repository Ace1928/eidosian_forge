import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import yaml
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.agent.agent import LaunchAgent
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from wandb.sdk.launch.registry.azure_container_registry import AzureContainerRegistry
from wandb.sdk.launch.registry.local_registry import LocalRegistry
from wandb.sdk.launch.runner.abstract import Status
from wandb.sdk.launch.runner.kubernetes_monitor import (
from wandb.util import get_module
from .._project_spec import EntryPoint, LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner
import kubernetes_asyncio  # type: ignore # noqa: E402
from kubernetes_asyncio import client  # noqa: E402
from kubernetes_asyncio.client.api.batch_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.core_v1_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.api.custom_objects_api import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.models.v1_secret import (  # type: ignore # noqa: E402
from kubernetes_asyncio.client.rest import ApiException  # type: ignore # noqa: E402
def get_namespace(self, resource_args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Get the namespace to launch into.

        Arguments:
            resource_args: The resource args to launch.
            context: The k8s config context.

        Returns:
            The namespace to launch into.
        """
    default_namespace = context['context'].get('namespace', 'default') if context else 'default'
    return resource_args.get('metadata', {}).get('namespace') or resource_args.get('namespace') or self.backend_config.get('runner', {}).get('namespace') or default_namespace