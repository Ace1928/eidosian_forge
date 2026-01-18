from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def create_sata_controller(bus_number):
    sata_ctl = vim.vm.device.VirtualDeviceSpec()
    sata_ctl.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    sata_ctl.device = vim.vm.device.VirtualAHCIController()
    sata_ctl.device.deviceInfo = vim.Description()
    sata_ctl.device.busNumber = bus_number
    sata_ctl.device.key = -randint(15000, 19999)
    return sata_ctl