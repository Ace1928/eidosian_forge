import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_certificate_authority(self, ca_perm: str, description: str=None, dry_run: bool=False):
    """
        Creates a Client Certificate Authority (CA).

        :param      ca_perm: The CA in PEM format. (required)
        :type       ca_perm: ``str``

        :param      description: The description of the CA.
        :type       description: ``bool``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: the created Ca.
        :rtype: ``dict``
        """
    action = 'CreateCa'
    data = {'DryRun': dry_run, 'CaPerm': ca_perm}
    if description is not None:
        data.update({'Description': description})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Ca']
    return response.json()