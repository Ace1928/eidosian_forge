import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_nic(self, description: str=None, link_nic_delete_on_vm_deletion: str=None, link_nic_id: str=None, security_group_ids: List[str]=None, nic_id: str=None, dry_run: bool=False):
    """
        Modifies the specified network interface card (NIC). You can specify
        only one attribute at a time.

        :param      description: A new description for the NIC.
        :type       description: ``str``

        :param      link_nic_delete_on_vm_deletion: If true, the NIC is
        deleted when the VM is terminated.
        :type       link_nic_delete_on_vm_deletion: ``str``

        :param      link_nic_id: The ID of the NIC attachment.
        :type       link_nic_id: ``str``

        :param      security_group_ids: One or more IDs of security groups
        for the NIC.
        :type       security_group_ids: ``list`` of ``str``

        :param      nic_id: The ID of the NIC you want to modify. (required)
        :type       nic_id: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Nic
        :rtype: ``dict``
        """
    action = 'UpdateNic'
    data = {'DryRun': dry_run, 'LinkNic': {}}
    if description is not None:
        data.update({'Description': description})
    if security_group_ids is not None:
        data.update({'SecurityGroupIds': security_group_ids})
    if nic_id is not None:
        data.update({'NicId': nic_id})
    if link_nic_delete_on_vm_deletion is not None:
        data['LinkNic'].update({'DeleteOnVmDeletion': link_nic_delete_on_vm_deletion})
    if link_nic_id is not None:
        data['LinkNic'].update({'LinkNicId': link_nic_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Nic']
    return response.json()