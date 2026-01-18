from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def create_nvme_controller(bus_number):
    nvme_ctl = vim.vm.device.VirtualDeviceSpec()
    nvme_ctl.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    nvme_ctl.device = vim.vm.device.VirtualNVMEController()
    nvme_ctl.device.deviceInfo = vim.Description()
    nvme_ctl.device.key = -randint(31000, 39999)
    nvme_ctl.device.busNumber = bus_number
    return nvme_ctl