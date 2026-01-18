import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_link_virtual_gateway(self, net_id: str=None, virtual_gateway_id: str=None, dry_run: bool=False):
    """
        Attaches a virtual gateway to a Net.

        :param      net_id: The ID of the Net to which you want to attach
        the virtual gateway. (required)
        :type       net_id: ``str``

        :param      virtual_gateway_id: The ID of the virtual
        gateway. (required)
        :type       virtual_gateway_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return:
        :rtype: ``dict``
        """
    action = 'LinkVirtualGateway'
    data = {'DryRun': dry_run}
    if net_id is not None:
        data.update({'NetId': net_id})
    if virtual_gateway_id is not None:
        data.update({'VirtualGatewayId': virtual_gateway_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetToVirtualGatewayLink']
    return response.json()