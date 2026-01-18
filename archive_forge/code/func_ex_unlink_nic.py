import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_unlink_nic(self, link_nic_id: str=None, dry_run: bool=False):
    """
        Detaches a network interface card (NIC) from a virtual machine (VM).
        The primary NIC cannot be detached.

        :param      link_nic_id: The ID of the NIC you want to delete.
        (required)
        :type       link_nic_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'UnlinkNic'
    data = {'DryRun': dry_run}
    if link_nic_id is not None:
        data.update({'LinkNicId': link_nic_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()