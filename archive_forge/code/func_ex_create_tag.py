import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_tag(self, resource_ids: list, tag_key: str=None, tag_value: str=None, dry_run: bool=False):
    """
        Adds one tag to the specified resources.
        If a tag with the same key already exists for the resource,
        the tag value is replaced.
        You can tag the following resources using their IDs:

        :param      resource_ids: One or more resource IDs. (required)
        :type       resource_ids: ``list``

        :param      tag_key: The key of the tag, with a minimum of 1 character.
        (required)
        :type       tag_key: ``str``

        :param      tag_value: The value of the tag, between
        0 and 255 characters. (required)
        :type       tag_value: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action. (required)
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'CreateTags'
    data = {'DryRun': dry_run, 'ResourceIds': resource_ids, 'Tags': []}
    if tag_key is not None and tag_value is not None:
        data['Tags'].append({'Key': tag_key, 'Value': tag_value})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()