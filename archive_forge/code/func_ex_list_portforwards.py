import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_portforwards(self, network):
    data = self._api_request('/portforwarding/list', {'cloudspaceId': int(network.id)})
    forwards = []
    for forward in data:
        forwards.append(self._to_port_forward(forward, network))
    return forwards