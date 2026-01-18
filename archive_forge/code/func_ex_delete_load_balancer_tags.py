import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_load_balancer_tags(self, load_balancer_names: List[str]=None, tag_keys: List[str]=None, dry_run: bool=False):
    """
        Deletes a specified load balancer tags.

        :param      load_balancer_names: The names of the load balancer for
        which you want to delete tags. (required)
        :type       load_balancer_names: ``str``

        :param      tag_keys: The key of the tag, with a minimum of
        1 character.
        :type       tag_keys: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteLoadBalancerTags'
    data = {'DryRun': dry_run, 'Tags': {}}
    if load_balancer_names is not None:
        data.update({'LoadBalancerNames': load_balancer_names})
    if tag_keys is not None:
        data['Tags'].update({'Keys': tag_keys})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()