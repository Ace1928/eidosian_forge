import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_virtual_gateway(self, virtual_gateway_id: str=None, dry_run: bool=False):
    """
        Deletes a specified virtual gateway.
        Before deleting a virtual gateway, we
        recommend to detach it from the Net and delete the VPN connection.

        :param      virtual_gateway_id: The ID of the virtual gateway
        you want to delete. (required)
        :type       virtual_gateway_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteVirtualGateway'
    data = {'DryRun': dry_run}
    if virtual_gateway_id is not None:
        data.update({'VirtualGatewayId': virtual_gateway_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()