import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_access_key(self, access_key_id: str=None, state: str=None, dry_run: bool=False):
    """
        Modifies the status of the specified access key associated with
        the account that sends the request.
        When set to ACTIVE, the access key is enabled and can be used to
        send requests. When set to INACTIVE, the access key is disabled.

        :param      access_key_id: The ID of the access key. (required)
        :type       access_key_id: ``str``

        :param      state: The new state of the access key
        (ACTIVE | INACTIVE). (required)
        :type       state: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Access Key
        :rtype: ``dict``
        """
    action = 'UpdateAccessKey'
    data = {'DryRun': dry_run}
    if access_key_id is not None:
        data.update({'AccessKeyId': access_key_id})
    if state is not None:
        data.update({'State': state})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['AccessKey']
    return response.json()