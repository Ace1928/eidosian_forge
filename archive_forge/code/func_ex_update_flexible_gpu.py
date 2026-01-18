import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_flexible_gpu(self, delete_on_vm_deletion: bool=None, flexible_gpu_id: str=None, dry_run: bool=False):
    """
        Modifies a flexible GPU (fGPU) behavior.

        :param      delete_on_vm_deletion: If true, the fGPU is deleted when
        the VM is terminated.
        :type       delete_on_vm_deletion: ``bool``

        :param      flexible_gpu_id: The ID of the fGPU you want to modify.
        :type       flexible_gpu_id: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: the updated Flexible GPU
        :rtype: ``dict``
        """
    action = 'UpdateFlexibleGpu'
    data = {'DryRun': dry_run, 'DirectLinkInterface': {}}
    if delete_on_vm_deletion is not None:
        data.update({'DeleteOnVmDeletion': delete_on_vm_deletion})
    if flexible_gpu_id is not None:
        data.update({'FlexibleGpuId': flexible_gpu_id})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['FlexibleGpu']
    return response.json()