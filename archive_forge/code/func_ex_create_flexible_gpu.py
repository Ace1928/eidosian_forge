import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_flexible_gpu(self, delete_on_vm_deletion: bool=None, generation: str=None, model_name: str=None, subregion_name: str=None, dry_run: bool=False):
    """
        Allocates a flexible GPU (fGPU) to your account.
        You can then attach this fGPU to a virtual machine (VM).

        :param      delete_on_vm_deletion: If true, the fGPU is deleted when
        the VM is terminated.
        :type       delete_on_vm_deletion: ``bool``

        :param      generation: The processor generation that the fGPU must be
        compatible with. If not specified, the oldest possible processor
        generation is selected (as provided by ReadFlexibleGpuCatalog for
        the specified model of fGPU).
        :type       generation: ``str``

        :param      model_name: The model of fGPU you want to allocate. For
        more information, see About Flexible GPUs:
        https://wiki.outscale.net/display/EN/About+Flexible+GPUs (required)
        :type       model_name: ``str``

        :param      subregion_name: The Subregion in which you want to create
        the fGPU. (required)
        :type       subregion_name: ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Flexible GPU
        :rtype: ``dict``
        """
    action = 'CreateFlexibleGpu'
    data = {'DryRun': dry_run}
    if delete_on_vm_deletion is not None:
        data.update({'DeleteOnVmDeletion': delete_on_vm_deletion})
    if generation is not None:
        data.update({'Generation': generation})
    if model_name is not None:
        data.update({'ModelName': model_name})
    if subregion_name is not None:
        data.update({'SubregionName': subregion_name})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['FlexibleGpu']
    return response.json()