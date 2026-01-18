import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_dhcp_options(self, dhcp_options_set_id: str=None, dry_run: bool=False):
    """
        Deletes a specified DHCP options set.
        Before deleting a DHCP options set, you must disassociate it from the
        Nets you associated it with. To do so, you need to associate with each
        Net a new set of DHCP options, or the default one if you do not want
        to associate any DHCP options with the Net.

        :param      dhcp_options_set_id: The ID of the DHCP options set
        you want to delete. (required)
        :type       dhcp_options_set_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteDhcpOptions'
    data = {'DryRun': dry_run}
    if dhcp_options_set_id is not None:
        data.update({'DhcpOptionsSetId': dhcp_options_set_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()