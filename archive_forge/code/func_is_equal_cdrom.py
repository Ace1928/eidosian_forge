from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def is_equal_cdrom(vm_obj, cdrom_device, cdrom_type, iso_path):
    if cdrom_type == 'none':
        return isinstance(cdrom_device.backing, vim.vm.device.VirtualCdrom.RemotePassthroughBackingInfo) and cdrom_device.connectable.allowGuestControl and (not cdrom_device.connectable.startConnected) and (vm_obj.runtime.powerState != vim.VirtualMachinePowerState.poweredOn or not cdrom_device.connectable.connected)
    elif cdrom_type == 'client':
        return isinstance(cdrom_device.backing, vim.vm.device.VirtualCdrom.RemotePassthroughBackingInfo) and cdrom_device.connectable.allowGuestControl and cdrom_device.connectable.startConnected and (vm_obj.runtime.powerState != vim.VirtualMachinePowerState.poweredOn or cdrom_device.connectable.connected)
    elif cdrom_type == 'iso':
        return isinstance(cdrom_device.backing, vim.vm.device.VirtualCdrom.IsoBackingInfo) and cdrom_device.backing.fileName == iso_path and cdrom_device.connectable.allowGuestControl and cdrom_device.connectable.startConnected and (vm_obj.runtime.powerState != vim.VirtualMachinePowerState.poweredOn or cdrom_device.connectable.connected)