import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_describe_address(self, ex_address_id, include=None):
    path = '/metal/v1/ips/%s' % ex_address_id
    params = {'include': include}
    result = self.connection.request(path, params=params).object
    return result