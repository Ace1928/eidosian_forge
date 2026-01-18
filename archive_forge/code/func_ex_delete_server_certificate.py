import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_server_certificate(self, name: str=None, dry_run: bool=False):
    """
        Deletes a specified server certificate.

        :param      name: The name of the server certificate you
        want to delete. (required)
        :type       name: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteServerCertificate'
    data = {'DryRun': dry_run}
    if name is not None:
        data.update({'Name': name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ResponseContext']
    return response.json()