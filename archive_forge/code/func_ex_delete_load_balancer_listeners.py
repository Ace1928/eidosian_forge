import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_load_balancer_listeners(self, load_balancer_name: str=None, load_balancer_ports: List[int]=None, dry_run: bool=False):
    """
        Deletes listeners of a specified load balancer.

        :param      load_balancer_name: The name of the load balancer for
        which you want to delete listeners. (required)
        :type       load_balancer_name: ``str``

        :param      load_balancer_ports: One or more port numbers of the
        listeners you want to delete.. (required)
        :type       load_balancer_ports: ``list`` of ``int``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeleteLoadBalancerListeners'
    data = {'DryRun': dry_run}
    if load_balancer_ports is not None:
        data.update({'LoadBalancerPorts': load_balancer_ports})
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()