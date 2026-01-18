import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_internet_service(self, dry_run: bool=False):
    """
        Creates an Internet service you can use with a Net.
        An Internet service enables your virtual machines (VMs) launched
        in a Net to connect to the Internet. By default, a Net includes
        an Internet service, and each Subnet is public. Every VM
        launched within a default Subnet has a private and a public IP
        addresses.

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Internet Service
        :rtype: ``dict``
        """
    action = 'CreateInternetService'
    data = {'DryRun': dry_run, 'DirectLinkInterface': {}}
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['InternetService']
    return response.json()