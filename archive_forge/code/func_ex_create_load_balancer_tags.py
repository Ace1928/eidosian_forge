import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_load_balancer_tags(self, load_balancer_names: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, dry_run: bool=False):
    """
        Adds one or more tags to the specified load balancers.
        If a tag with the same key already exists for the load balancer,
        the tag value is replaced.

        :param      load_balancer_names: The name of the load balancer for
        which you want to create listeners. (required)
        :type       load_balancer_names: ``str``

        :param      tag_keys: The key of the tag, with a minimum of 1
        character. (required)
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The value of the tag, between 0 and 255
        characters. (required)
        :type       tag_values: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Load Balancer Tags
        :rtype: ``dict``
        """
    action = 'CreateLoadBalancerTags'
    data = {'DryRun': dry_run, 'Tags': {}}
    if load_balancer_names is not None:
        data.update({'LoadBalancerNames': load_balancer_names})
    if tag_keys and tag_values and (len(tag_keys) == len(tag_values)):
        for key, value in zip(tag_keys, tag_values):
            data['Tags'].update({'Key': key, 'Value': value})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['LoadBalancer']
    return response.json()