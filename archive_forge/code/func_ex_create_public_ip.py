import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_public_ip(self, dry_run: bool=False):
    """
        Create a new public ip.

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: the created public ip
        :rtype: ``dict``
        """
    action = 'CreatePublicIp'
    data = json.dumps({'DryRun': dry_run})
    response = self._call_api(action, data)
    if response.status_code == 200:
        return response.json()['PublicIp']
    return response.json()