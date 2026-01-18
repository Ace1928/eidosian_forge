import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_enable_ipv6(self, node):
    attr = {'type': 'enable_ipv6'}
    res = self.connection.request('/v2/droplets/%s/actions' % node.id, data=json.dumps(attr), method='POST')
    return res.status == httplib.CREATED