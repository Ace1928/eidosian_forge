import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_virtual_gateway(self, connection_type: str=None, dry_run: bool=False):
    """
        Creates a virtual gateway.
        A virtual gateway is the access point on the Net
        side of a VPN connection.

        :param      connection_type: The type of VPN connection supported
        by the virtual gateway (only ipsec.1 is supported). (required)
        :type       connection_type: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new virtual gateway
        :rtype: ``dict``
        """
    action = 'CreateVirtualGateway'
    data = {'DryRun': dry_run}
    if connection_type is not None:
        data.update({'ConnectionType': connection_type})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['VirtualGateway']
    return response.json()