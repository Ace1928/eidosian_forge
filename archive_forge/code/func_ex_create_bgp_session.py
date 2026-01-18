import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_create_bgp_session(self, node, address_family='ipv4'):
    path = '/metal/v1/devices/%s/bgp/sessions' % node.id
    params = {'address_family': address_family}
    res = self.connection.request(path, params=params, method='POST')
    return res.object