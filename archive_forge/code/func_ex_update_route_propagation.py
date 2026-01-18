import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_route_propagation(self, enable: bool=None, route_table_id: str=None, virtual_gateway_id: str=None, dry_run: bool=False):
    """
        Configures the propagation of routes to a specified route table
        of a Net by a virtual gateway.

        :param      enable: If true, a virtual gateway can propagate routes
        to a specified route table of a Net. If false,
        the propagation is disabled. (required)
        :type       enable: ``boolean``

        :param      route_table_id: The ID of the route table. (required)
        :type       route_table_id: ``str``

        :param      virtual_gateway_id: The ID of the virtual
        gateway. (required)
        :type       virtual_gateway_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: route propagation
        :rtype: ``dict``
        """
    action = 'UpdateRoutePropagation'
    data = {'DryRun': dry_run}
    if enable is not None:
        data.update({'Enable': enable})
    if route_table_id is not None:
        data.update({'RouteTableId': route_table_id})
    if virtual_gateway_id is not None:
        data.update({'VirtualGatewayId': virtual_gateway_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['RouteTable']
    return response.json()