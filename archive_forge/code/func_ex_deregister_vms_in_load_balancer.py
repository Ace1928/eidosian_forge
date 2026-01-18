import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_deregister_vms_in_load_balancer(self, backend_vm_ids: List[str]=None, load_balancer_name: str=None, dry_run: bool=False):
    """
        Deregisters a specified virtual machine (VM) from a load balancer.

        :param      backend_vm_ids: One or more IDs of back-end VMs.
        (required)
        :type       backend_vm_ids: ``str``

        :param      load_balancer_name: The name of the load balancer.
        (required)
        :type       load_balancer_name: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'DeregisterVmsInLoadBalancer'
    data = {'DryRun': dry_run}
    if load_balancer_name is not None:
        data.update({'LoadBalancerName': load_balancer_name})
    if backend_vm_ids is not None:
        data.update({'BackendVmIds': backend_vm_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()