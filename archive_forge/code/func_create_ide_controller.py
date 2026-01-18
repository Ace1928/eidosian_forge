from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def create_ide_controller(bus_number=0):
    ide_ctl = vim.vm.device.VirtualDeviceSpec()
    ide_ctl.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    ide_ctl.device = vim.vm.device.VirtualIDEController()
    ide_ctl.device.deviceInfo = vim.Description()
    ide_ctl.device.key = -randint(200, 299)
    ide_ctl.device.busNumber = bus_number
    return ide_ctl