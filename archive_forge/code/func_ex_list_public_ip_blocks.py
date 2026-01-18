import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_public_ip_blocks(self, network_domain):
    params = {}
    params['networkDomainId'] = network_domain.id
    response = self.connection.request_with_orgId_api_2('network/publicIpBlock', params=params).object
    return self._to_ip_blocks(response)