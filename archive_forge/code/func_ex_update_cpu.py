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
def ex_update_cpu(self, node, cores):
    """
        Assuming 1 Core per socket
        :param cores: Integer or string indicating number of cores
        :type cores: `int` or `str`
        """
    if isinstance(node, str):
        node_id = node
    else:
        node_id = node.id
    request = '/rest/vcenter/vm/{}/hardware/cpu'.format(node_id)
    cores = int(cores)
    body = {'spec': {'count': cores}}
    response = self._request(request, method='PATCH', data=json.dumps(body))
    return response.status in self.VALID_RESPONSE_CODES