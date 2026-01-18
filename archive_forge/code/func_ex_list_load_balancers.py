import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_load_balancers(self, load_balancer_names: List[str]=None, dry_run: bool=False):
    """
        Lists one or more load balancers and their attributes.

        :param      load_balancer_names: The names of the load balancer.
        (required)
        :type       load_balancer_names: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of load balancer
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadLoadBalancers'
    data = {'DryRun': dry_run, 'Filters': {}}
    if load_balancer_names is not None:
        data['Filters'].update({'LoadBalancerNames': load_balancer_names})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['LoadBalancers']
    return response.json()