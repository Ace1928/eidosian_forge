import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_destroy_network(self, network):
    self._api_request('/cloudspaces/delete', {'cloudspaceId': int(network.id)})
    return True