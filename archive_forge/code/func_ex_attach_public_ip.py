import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_attach_public_ip(self, allow_relink: bool=None, dry_run: bool=False, nic_id: str=None, vm_id: str=None, public_ip: str=None, public_ip_id: str=None):
    """
        Attach public ip to a node.

        :param      allow_relink: If true, allows the EIP to be associated
        with the VM or NIC that you specify even if
        it is already associated with another VM or NIC.
        :type       allow_relink: ``bool``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :param      nic_id:(Net only) The ID of the NIC. This parameter is
        required if the VM has more than one NIC attached. Otherwise,
        you need to specify the VmId parameter instead.
        You cannot specify both parameters
        at the same time.
        :type       nic_id: ``str``

        :param      vm_id: the ID of the VM
        :type       nic_id: ``str``

        :param      public_ip: The EIP. In the public Cloud, this parameter
        is required.
        :type       public_ip: ``str``

        :param      public_ip_id: The allocation ID of the EIP. In a Net,
        this parameter is required.
        :type       public_ip_id: ``str``

        :return: the attached volume
        :rtype: ``dict``
        """
    action = 'LinkPublicIp'
    data = {'DryRun': dry_run}
    if public_ip is not None:
        data.update({'PublicIp': public_ip})
    if public_ip_id is not None:
        data.update({'PublicIpId': public_ip_id})
    if nic_id is not None:
        data.update({'NicId': nic_id})
    if vm_id is not None:
        data.update({'VmId': vm_id})
    if allow_relink is not None:
        data.update({'AllowRelink': allow_relink})
    data = json.dumps(data)
    response = self._call_api(action, data)
    if response.status_code == 200:
        return True
    return response.json()