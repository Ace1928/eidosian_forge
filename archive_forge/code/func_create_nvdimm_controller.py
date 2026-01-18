from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def create_nvdimm_controller(self):
    nvdimm_ctl = vim.vm.device.VirtualDeviceSpec()
    nvdimm_ctl.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    nvdimm_ctl.device = vim.vm.device.VirtualNVDIMMController()
    nvdimm_ctl.device.deviceInfo = vim.Description()
    nvdimm_ctl.device.key = -randint(27000, 27999)
    return nvdimm_ctl