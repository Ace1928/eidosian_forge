import json
import logging
from pathlib import Path
from threading import RLock
from uuid import uuid4
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
from ray.autoscaler._private._azure.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def stopped_nodes(self, tag_filters):
    """Return a list of stopped node ids filtered by the specified tags dict."""
    nodes = self._get_filtered_nodes(tag_filters=tag_filters)
    return [k for k, v in nodes.items() if v['status'].startswith('deallocat')]