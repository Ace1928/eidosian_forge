import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_public_ip(self, dry_run: bool=False, public_ip: str=None, public_ip_id: str=None):
    """
        Delete public ip.

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :param      public_ip: The EIP. In the public Cloud, this parameter is
        required.
        :type       public_ip: ``str``

        :param      public_ip_id: The ID representing the association of the
        EIP with the VM or the NIC. In a Net,
        this parameter is required.
        :type       public_ip_id: ``str``

        :return: request
        :rtype: ``dict``
        """
    action = 'DeletePublicIp'
    data = {'DryRun': dry_run}
    if public_ip is not None:
        data.update({'PublicIp': public_ip})
    if public_ip_id is not None:
        data.update({'PublicIpId': public_ip_id})
    data = json.dumps(data)
    response = self._call_api(action, data)
    if response.status_code == 200:
        return True
    return response.json()