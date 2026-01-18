from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def create_nvdimm_device(self, nvdimm_ctl_dev_key, pmem_profile_id, nvdimm_dev_size_mb=1024):
    nvdimm_dev_spec = vim.vm.device.VirtualDeviceSpec()
    nvdimm_dev_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    nvdimm_dev_spec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.create
    nvdimm_dev_spec.device = vim.vm.device.VirtualNVDIMM()
    nvdimm_dev_spec.device.controllerKey = nvdimm_ctl_dev_key
    nvdimm_dev_spec.device.key = -randint(28000, 28999)
    nvdimm_dev_spec.device.capacityInMB = nvdimm_dev_size_mb
    nvdimm_dev_spec.device.deviceInfo = vim.Description()
    nvdimm_dev_spec.device.backing = vim.vm.device.VirtualNVDIMM.BackingInfo()
    if pmem_profile_id is not None:
        profile = vim.vm.DefinedProfileSpec()
        profile.profileId = pmem_profile_id
        nvdimm_dev_spec.profile = [profile]
    return nvdimm_dev_spec