import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_list_ip_assignments_for_node(self, node, include=''):
    path = '/metal/v1/devices/%s/ips' % node.id
    params = {'include': include}
    return self.connection.request(path, params=params).object