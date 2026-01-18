import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_tags(self, resource_ids: list=None, resource_types: list=None, keys: list=None, values: list=None, dry_run: bool=False):
    """
        Lists one or more tags for your resources.

        :param      resource_ids: One or more resource IDs.
        :type       resource_ids: ``list``

        :param      resource_types: One or more resource IDs.
        :type       resource_types: ``list``

        :param      keys: One or more resource IDs.
        :type       keys: ``list``

        :param      values: One or more resource IDs.
        :type       values: ``list``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: list of tags
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadTags'
    data = {'Filters': {}, 'DryRun': dry_run}
    if resource_ids is not None:
        data['Filters'].update({'ResourceIds': resource_ids})
    if resource_types is not None:
        data['Filters'].update({'ResourceTypes': resource_types})
    if keys is not None:
        data['Filters'].update({'Keys': keys})
    if values is not None:
        data['Filters'].update({'Values': values})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Tags']
    return response.json()