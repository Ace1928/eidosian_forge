import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_get_node_bandwidth(self, node, from_time, until_time):
    path = '/metal/v1/devices/%s/bandwidth' % node.id
    params = {'from': from_time, 'until': until_time}
    return self.connection.request(path, params=params).object