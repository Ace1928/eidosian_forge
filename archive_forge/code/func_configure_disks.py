from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def configure_disks(self, vm_obj):
    if not self.params['disk']:
        return
    configure_multiple_ctl = False
    for disk_spec in self.params.get('disk'):
        if disk_spec['controller_type'] or disk_spec['controller_number'] or disk_spec['unit_number']:
            configure_multiple_ctl = True
            break
    if configure_multiple_ctl:
        self.configure_multiple_controllers_disks(vm_obj)
        return
    scsi_ctls = self.get_vm_scsi_controllers(vm_obj)
    if vm_obj is None or not scsi_ctls:
        scsi_ctl = self.device_helper.create_scsi_controller(self.get_scsi_type(), 0)
        self.change_detected = True
        self.configspec.deviceChange.append(scsi_ctl)
    else:
        scsi_ctl = scsi_ctls[0]
    disks = [x for x in vm_obj.config.hardware.device if isinstance(x, vim.vm.device.VirtualDisk)] if vm_obj is not None else None
    if disks is not None and self.params.get('disk') and (len(self.params.get('disk')) < len(disks)):
        self.module.fail_json(msg='Provided disks configuration has less disks than the target object (%d vs %d)' % (len(self.params.get('disk')), len(disks)))
    disk_index = 0
    for expected_disk_spec in self.params.get('disk'):
        disk_modified = False
        if vm_obj is not None and disks is not None and (disk_index < len(disks)):
            diskspec = vim.vm.device.VirtualDeviceSpec()
            diskspec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
            diskspec.device = disks[disk_index]
        else:
            diskspec = self.device_helper.create_hard_disk(scsi_ctl, disk_index)
            disk_modified = True
        disk_index += 1
        if disk_index == 7:
            disk_index += 1
        if expected_disk_spec['disk_mode']:
            disk_mode = expected_disk_spec.get('disk_mode', 'persistent')
            if vm_obj and diskspec.device.backing.diskMode != disk_mode or vm_obj is None:
                diskspec.device.backing.diskMode = disk_mode
                disk_modified = True
        else:
            diskspec.device.backing.diskMode = 'persistent'
        if expected_disk_spec['type']:
            disk_type = expected_disk_spec.get('type', '').lower()
            if disk_type == 'thin':
                diskspec.device.backing.thinProvisioned = True
            elif disk_type == 'eagerzeroedthick':
                diskspec.device.backing.eagerlyScrub = True
        if expected_disk_spec['filename']:
            self.add_existing_vmdk(vm_obj, expected_disk_spec, diskspec, scsi_ctl)
            continue
        if vm_obj is None or self.params['template']:
            if diskspec.device.backing.fileName == '':
                diskspec.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.create
        if expected_disk_spec.get('datastore'):
            pass
        kb = self.get_configured_disk_size(expected_disk_spec)
        if kb < diskspec.device.capacityInKB:
            self.module.fail_json(msg='Given disk size is smaller than found (%d < %d). Reducing disks is not allowed.' % (kb, diskspec.device.capacityInKB))
        if kb != diskspec.device.capacityInKB or disk_modified:
            diskspec.device.capacityInKB = kb
            self.configspec.deviceChange.append(diskspec)
            self.change_detected = True