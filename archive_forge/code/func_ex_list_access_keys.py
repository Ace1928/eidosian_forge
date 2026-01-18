import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_access_keys(self, access_key_ids: list=None, states: list=None, dry_run: bool=False):
    """
        Returns information about the access key IDs of a specified user.
        If the user does not have any access key ID, this action returns
        an empty list.

        :param      access_key_ids: The IDs of the access keys.
        :type       access_key_ids: ``list`` of ``str``

        :param      states: The states of the access keys (ACTIVE | INACTIVE).
        :type       states: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: ``list`` of Access Keys
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadAccessKeys'
    data = {'DryRun': dry_run, 'Filters': {}}
    if access_key_ids is not None:
        data['Filters'].update({'AccessKeyIds': access_key_ids})
    if states is not None:
        data['Filters'].update({'States': states})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['AccessKeys']
    return response.json()