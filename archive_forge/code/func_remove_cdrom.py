from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def remove_cdrom(self, cdrom_device):
    cdrom_spec = vim.vm.device.VirtualDeviceSpec()
    cdrom_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
    cdrom_spec.device = cdrom_device
    return cdrom_spec