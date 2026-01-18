import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def stop_node(self, node):
    """
        Stop virtual machine
        """
    node.state = NodeState.STOPPING
    self._api_request('/machines/stop', {'machineId': int(node.id)})
    node.state = NodeState.STOPPED
    return True