import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_vpn_connection(self, client_gateway_id: str=None, connection_type: str=None, static_routes_only: bool=None, virtual_gateway_id: str=None, dry_run: bool=False):
    """
        Creates a VPN connection between a specified virtual gateway and a
        specified client gateway.
        You can create only one VPN connection between a virtual gateway and
        a client gateway.

        :param      client_gateway_id: The ID of the client gateway. (required)
        :type       client_gateway_id: ``str``

        :param      connection_type: The type of VPN connection (only ipsec.1
        is supported). (required)
        :type       connection_type: ``str``

        :param      static_routes_only: If false, the VPN connection uses
        dynamic routing with Border Gateway Protocol (BGP). If true, routing
        is controlled using static routes. For more information about how to
        create and delete static routes, see CreateVpnConnectionRoute:
        https://docs.outscale.com/api#createvpnconnectionroute and
        DeleteVpnConnectionRoute:
        https://docs.outscale.com/api#deletevpnconnectionroute
        :type       static_routes_only: ``bool``

        :param      virtual_gateway_id: The ID of the virtual gateway.
        (required)
        :type       virtual_gateway_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Vpn Connection
        :rtype: ``dict``
        """
    action = 'CreateVpnConnection'
    data = {'DryRun': dry_run}
    if client_gateway_id is not None:
        data.update({'ClientGatewayId': client_gateway_id})
    if connection_type is not None:
        data.update({'ConnectionType': connection_type})
    if static_routes_only is not None:
        data.update({'StaticRoutesOnly': static_routes_only})
    if virtual_gateway_id is not None:
        data.update({'StaticRoutesOnly': virtual_gateway_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['VpnConnection']
    return response.json()