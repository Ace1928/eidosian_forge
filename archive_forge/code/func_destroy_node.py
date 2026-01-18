import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def destroy_node(self, node):
    """
        Destroy node
        """
    self._api_request('/machines/delete', {'machineId': int(node.id)})
    return True