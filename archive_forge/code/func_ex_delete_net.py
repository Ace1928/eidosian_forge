import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_net(self, net_id: str=None, dry_run: bool=False):
    """
        Deletes a specified Net.
        Before deleting the Net, you need to delete or detach all the
        resources associated with the Net:

        - Virtual machines (VMs)
        - Net peering connections
        - Custom route tables
        - External IP addresses (EIPs) allocated to resources in the Net
        - Network Interface Cards (NICs) created in the Subnets
        - Virtual gateways, Internet services and NAT services
        - Load balancers
        - Security groups
        - Subnets

        :param      net_id: The ID of the Net you want to delete. (required)
        :type       net_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteNet'
    data = {'DryRun': dry_run}
    if net_id is not None:
        data.update({'NetId': net_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()