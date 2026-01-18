from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def remove_nvdimm(self, nvdimm_device):
    nvdimm_spec = vim.vm.device.VirtualDeviceSpec()
    nvdimm_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
    nvdimm_spec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.destroy
    nvdimm_spec.device = nvdimm_device
    return nvdimm_spec