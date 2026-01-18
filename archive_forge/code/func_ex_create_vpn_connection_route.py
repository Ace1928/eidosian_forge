import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_vpn_connection_route(self, destination_ip_range: str=None, vpn_connection_id: str=None, dry_run: bool=False):
    """
        Creates a static route to a VPN connection.
        This enables you to select the network flows sent by the virtual
        gateway to the target VPN connection.

        :param      destination_ip_range: The network prefix of the route, in
        CIDR notation (for example, 10.12.0.0/16).(required)
        :type       destination_ip_range: ``str``

        :param      vpn_connection_id: The ID of the target VPN connection of
        the static route. (required)
        :type       vpn_connection_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'CreateVpnConnectionRoute'
    data = {'DryRun': dry_run}
    if destination_ip_range is not None:
        data.update({'DestinationIpRange': destination_ip_range})
    if vpn_connection_id is not None:
        data.update({'VpnConnectionId': vpn_connection_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()