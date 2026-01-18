import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_unlink_internet_service(self, internet_service_id: str=None, net_id: str=None, dry_run: bool=False):
    """
        Detaches an Internet service from a Net.
        This action disables and detaches an Internet service from a Net.
        The Net must not contain any running virtual machine (VM) using an
        External IP address (EIP).

        :param      internet_service_id: The ID of the Internet service you
        want to detach. (required)
        :type       internet_service_id: ``str``

        :param      net_id: The ID of the Net from which you want to detach
        the Internet service. (required)
        :type       net_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'UnlinkInternetService'
    data = {'DryRun': dry_run}
    if internet_service_id is not None:
        data.update({'InternetServiceId': internet_service_id})
    if net_id is not None:
        data.update({'NetId': net_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()