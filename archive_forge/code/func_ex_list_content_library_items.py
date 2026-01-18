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
def ex_list_content_library_items(self, library_id):
    req = '/rest/com/vmware/content/library/item'
    params = {'library_id': library_id}
    try:
        result = self._request(req, params=params).object
        return result['value']
    except BaseHTTPError:
        logger.error('Library was cannot be accessed,  most probably the VCenter service is stopped')
        return []