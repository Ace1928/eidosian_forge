import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_list_bgp_sessions_for_node(self, node):
    path = '/metal/v1/devices/%s/bgp/sessions' % node.id
    return self.connection.request(path).object