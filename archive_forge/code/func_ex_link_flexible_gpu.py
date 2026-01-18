import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_link_flexible_gpu(self, flexible_gpu_id: str=None, vm_id: str=None, dry_run: bool=False):
    """
        Attaches one of your allocated flexible GPUs (fGPUs) to one of your
        virtual machines (Nodes).
        The fGPU is in the attaching state until the VM is stopped, after
        which it becomes attached.

        :param      flexible_gpu_id: The ID of the fGPU you want to attach.
        (required)
        :type       flexible_gpu_id: ``str``

        :param      vm_id: The ID of the VM you want to attach the fGPU to.
        (required)
        :type       vm_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: True if the action is successful
        :rtype: ``bool``
        """
    action = 'LinkFlexibleGpu'
    data = {'DryRun': dry_run}
    if flexible_gpu_id is not None:
        data.update({'FlexibleGpuId': flexible_gpu_id})
    if vm_id is not None:
        data.update({'VmId': vm_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return True
    return response.json()