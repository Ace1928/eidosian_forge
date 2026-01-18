import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_reinstall_node(self, node):
    params = {'type': 'reinstall'}
    res = self.connection.request('/metal/v1/devices/%s/actions' % node.id, params=params, method='POST')
    return res.status == httplib.OK