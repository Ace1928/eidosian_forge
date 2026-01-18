import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_direct_link_interface(self, direct_link_interface_id: str=None, dry_run: bool=False):
    """
        Deletes a specified DirectLink interface.

        :param      direct_link_interface_id: the ID of the DirectLink
        interface you want to delete. (required)
        :type       direct_link_interface_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteDirectLinkInterface'
    data = {'DryRun': dry_run}
    if direct_link_interface_id is not None:
        data.update({'DirectLinkInterfaceId': direct_link_interface_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()