import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def list_nodes_recursive(self, enhance=True):
    """
        Lists nodes, excluding templates
        """
    nodes = []
    content = self.connection.RetrieveContent()
    children = content.rootFolder.childEntity
    if content.customFieldsManager:
        self.custom_fields = content.customFieldsManager.field
    else:
        self.custom_fields = []
    for child in children:
        if hasattr(child, 'vmFolder'):
            datacenter = child
            vm_folder = datacenter.vmFolder
            vm_list = vm_folder.childEntity
            nodes.extend(self._to_nodes_recursive(vm_list))
    if enhance:
        nodes = self._enhance_metadata(nodes, content)
    return nodes