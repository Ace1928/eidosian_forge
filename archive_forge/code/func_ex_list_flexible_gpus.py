import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_flexible_gpus(self, delete_on_vm_deletion: bool=None, flexible_gpu_ids: list=None, generations: list=None, model_names: list=None, states: list=None, subregion_names: list=None, vm_ids: list=None, dry_run: bool=False):
    """
        Lists one or more flexible GPUs (fGPUs) allocated to your account.

        :param      delete_on_vm_deletion: Indicates whether the fGPU is
        deleted when terminating the VM.
        :type       delete_on_vm_deletion: ``bool``

        :param      flexible_gpu_ids: One or more IDs of fGPUs.
        :type       flexible_gpu_ids: ``list`` of ``str``

        :param      generations: The processor generations that the fGPUs are
        compatible with.
        (required)
        :type       generations: ``list`` of ``str``

        :param      model_names: One or more models of fGPUs. For more
        information, see About Flexible GPUs:
        https://wiki.outscale.net/display/EN/About+Flexible+GPUs
        :type       model_names: ``list`` of ``str``

        :param      states: The states of the fGPUs
        (allocated | attaching | attached | detaching).
        :type       states: ``list`` of ``str``

        :param      subregion_names: The Subregions where the fGPUs are
        located.
        :type       subregion_names: ``list`` of ``str``

        :param      vm_ids: One or more IDs of VMs.
        :type       vm_ids: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: Returns the Flexible Gpu Catalog
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadFlexibleGpus'
    data = {'DryRun': dry_run, 'Filters': {}}
    if delete_on_vm_deletion is not None:
        data['Filters'].update({'DeleteOnVmDeletion': delete_on_vm_deletion})
    if flexible_gpu_ids is not None:
        data['Filters'].update({'FlexibleGpuIds': flexible_gpu_ids})
    if generations is not None:
        data['Filters'].update({'Generations': generations})
    if model_names is not None:
        data['Filters'].update({'ModelNames': model_names})
    if states is not None:
        data['Filters'].update({'States': states})
    if subregion_names is not None:
        data['Filters'].update({'SubregionNames': subregion_names})
    if vm_ids is not None:
        data['Filters'].update({'VmIds': vm_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['FlexibleGpus']
    return response.json()