import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_request_address_reservation(self, ex_project_id, location_id=None, address_family='global_ipv4', quantity=1, comments='', customdata=''):
    path = '/metal/v1/projects/%s/ips' % ex_project_id
    params = {'type': address_family, 'quantity': quantity}
    if location_id:
        params['facility'] = location_id
    if comments:
        params['comments'] = comments
    if customdata:
        params['customdata'] = customdata
    result = self.connection.request(path, params=params, method='POST').object
    return result