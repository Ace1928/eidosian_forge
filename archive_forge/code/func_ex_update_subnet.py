import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_subnet(self, subnet_id: str=None, map_public_ip_on_launch: bool=None, dry_run: bool=False):
    """
        Deletes a specified Subnet.
        You must terminate all the running virtual machines (VMs) in the
        Subnet before deleting it.

        :param      subnet_id: The ID of the Subnet you want to delete.
        (required)
        :type       subnet_id: ``str``

        :param      map_public_ip_on_launch: If true, a public IP address is
        assigned to the network interface cards (NICs) created in the s
        specified Subnet. (required)
        :type       map_public_ip_on_launch: ``bool``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The updated Subnet
        :rtype: ``dict``
        """
    action = 'UpdateSubnet'
    data = {'DryRun': dry_run}
    if subnet_id is not None:
        data.update({'SubnetId': subnet_id})
    if map_public_ip_on_launch is not None:
        data.update({'MapPublicIpOnLaunch': map_public_ip_on_launch})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Subnet']
    return response.json()