import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_link_nic(self, device_number: int=None, nic_id: str=None, node: str=None, dry_run: bool=False):
    """
        Attaches a network interface card (NIC) to a virtual machine (VM).
        The interface and the VM must be in the same Subregion. The VM can be
        either running or stopped. The NIC must be in the available state.

        :param      nic_id: The ID of the NIC you want to delete. (required)
        :type       nic_id: ``str``

        :param      device_number: The ID of the NIC you want to delete.
        (required)
        :type       device_number: ``str``

        :param      node: The index of the VM device for the NIC attachment
        (between 1 and 7, both included).
        :type       node: ``Node``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a Link Id
        :rtype: ``str``
        """
    action = 'LinkNic'
    data = {'DryRun': dry_run}
    if nic_id is not None:
        data.update({'NicId': nic_id})
    if device_number is not None:
        data.update({'DeviceNumber': device_number})
    if node:
        data.update({'VmId': node})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['LinkNicId']
    return response.json()