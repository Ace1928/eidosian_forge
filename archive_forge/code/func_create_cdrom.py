from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def create_cdrom(ctl_device, cdrom_type, iso_path=None, unit_number=0):
    cdrom_spec = vim.vm.device.VirtualDeviceSpec()
    cdrom_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    cdrom_spec.device = vim.vm.device.VirtualCdrom()
    cdrom_spec.device.controllerKey = ctl_device.key
    if isinstance(ctl_device, vim.vm.device.VirtualIDEController):
        cdrom_spec.device.key = -randint(3000, 3999)
    elif isinstance(ctl_device, vim.vm.device.VirtualAHCIController):
        cdrom_spec.device.key = -randint(16000, 16999)
    cdrom_spec.device.unitNumber = unit_number
    cdrom_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
    cdrom_spec.device.connectable.allowGuestControl = True
    cdrom_spec.device.connectable.startConnected = cdrom_type != 'none'
    if cdrom_type in ['none', 'client']:
        cdrom_spec.device.backing = vim.vm.device.VirtualCdrom.RemotePassthroughBackingInfo()
    elif cdrom_type == 'iso':
        cdrom_spec.device.backing = vim.vm.device.VirtualCdrom.IsoBackingInfo(fileName=iso_path)
        cdrom_spec.device.connectable.connected = True
    return cdrom_spec