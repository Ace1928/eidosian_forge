import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def reboot_node(self, node):
    """
        Reboot node
        returns True as if the reboot had been successful.
        """
    node.state = NodeState.REBOOTING
    self._api_request('/machines/reboot', {'machineId': int(node.id)})
    node.state = NodeState.RUNNING
    return True