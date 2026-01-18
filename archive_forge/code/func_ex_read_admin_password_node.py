import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_read_admin_password_node(self, node: Node, dry_run: bool=False):
    """
        Retrieves the administrator password for a Windows running virtual
        machine (VM).
        The administrator password is encrypted using the keypair you
        specified when launching the VM.

        :param      node: the ID of the VM (required)
        :type       node: ``Node``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The Admin Password of the specified Node.
        :rtype: ``str``
        """
    action = 'ReadAdminPassword'
    data = json.dumps({'DryRun': dry_run, 'VmId': node.id})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return response.json()['AdminPassword']
    return response.json()