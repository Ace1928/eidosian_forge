from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def create_hard_disk(self, disk_ctl, disk_index=None):
    diskspec = vim.vm.device.VirtualDeviceSpec()
    diskspec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    diskspec.device = vim.vm.device.VirtualDisk()
    diskspec.device.key = -randint(20000, 24999)
    diskspec.device.backing = vim.vm.device.VirtualDisk.FlatVer2BackingInfo()
    diskspec.device.controllerKey = disk_ctl.device.key
    if self.is_scsi_controller(disk_ctl.device):
        if disk_index is None:
            self.module.fail_json(msg='unitNumber for sata disk is None.')
        elif disk_index == 7 or disk_index > 15:
            self.module.fail_json(msg='Invalid scsi disk unitNumber, valid 0-15(except 7).')
        else:
            diskspec.device.unitNumber = disk_index
    elif self.is_sata_controller(disk_ctl.device):
        if disk_index is None:
            self.module.fail_json(msg='unitNumber for sata disk is None.')
        elif disk_index > 29:
            self.module.fail_json(msg='Invalid sata disk unitNumber, valid 0-29.')
        else:
            diskspec.device.unitNumber = disk_index
    elif self.is_nvme_controller(disk_ctl.device):
        if disk_index is None:
            self.module.fail_json(msg='unitNumber for nvme disk is None.')
        elif disk_index > 14:
            self.module.fail_json(msg='Invalid nvme disk unitNumber, valid 0-14.')
        else:
            diskspec.device.unitNumber = disk_index
    return diskspec