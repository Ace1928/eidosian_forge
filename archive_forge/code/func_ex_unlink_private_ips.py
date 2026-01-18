import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_unlink_private_ips(self, nic_id: str=None, private_ips: List[str]=None, dry_run: bool=False):
    """
        Unassigns one or more secondary private IPs from a network interface
        card (NIC).

        :param      nic_id: The ID of the NIC. (required)
        :type       nic_id: ``str``

        :param      private_ips: One or more secondary private IP addresses
        you want to unassign from the NIC. (required)
        :type       private_ips: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'UnlinkPrivateIps'
    data = {'DryRun': dry_run}
    if nic_id is not None:
        data.update({'NicId': nic_id})
    if private_ips is not None:
        data.update({'PrivateIps': private_ips})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()