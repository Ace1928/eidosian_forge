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
def configure_multiple_controllers_disks(self, vm_obj):
    ctls = self.sanitize_disk_parameters(vm_obj)
    if len(ctls) == 0:
        return
    for ctl in ctls:
        disk_ctl, disk_list = self.device_helper.get_controller_disks(vm_obj, ctl['type'], ctl['num'])
        if disk_ctl is None:
            if ctl['type'] in self.device_helper.scsi_device_type.keys() and vm_obj is not None:
                scsi_ctls = self.get_vm_scsi_controllers(vm_obj)
                if scsi_ctls:
                    for scsi_ctl in scsi_ctls:
                        if scsi_ctl.device.busNumber == ctl['num']:
                            self.module.fail_json(msg="Specified SCSI controller number '%s' is already used by: %s" % (ctl['num'], scsi_ctl))
            disk_ctl_spec = self.device_helper.create_disk_controller(ctl['type'], ctl['num'])
            self.change_detected = True
            self.configspec.deviceChange.append(disk_ctl_spec)
        else:
            disk_ctl_spec = vim.vm.device.VirtualDeviceSpec()
            disk_ctl_spec.device = disk_ctl
        for j in range(0, len(ctl['disk'])):
            hard_disk = None
            hard_disk_spec = None
            hard_disk_exist = False
            disk_modified_for_spec = False
            disk_modified_for_disk = False
            disk_unit_number = ctl['disk'][j]['unit_number']
            if len(disk_list) != 0:
                for disk in disk_list:
                    if disk.unitNumber == disk_unit_number:
                        hard_disk = disk
                        hard_disk_exist = True
                        break
            if hard_disk_exist:
                hard_disk_spec = vim.vm.device.VirtualDeviceSpec()
                hard_disk_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
                hard_disk_spec.device = hard_disk
                disk_modified_for_spec = self.set_disk_parameters(hard_disk_spec, ctl['disk'][j], reconfigure=True)
            if len(disk_list) == 0 or not hard_disk_exist:
                hard_disk = self.device_helper.create_hard_disk(disk_ctl_spec, disk_unit_number)
                hard_disk.fileOperation = vim.vm.device.VirtualDeviceSpec.FileOperation.create
                disk_modified_for_disk = self.set_disk_parameters(hard_disk, ctl['disk'][j])
            if disk_modified_for_spec:
                self.change_detected = True
                self.configspec.deviceChange.append(hard_disk_spec)
            if disk_modified_for_disk:
                self.change_detected = True
                self.configspec.deviceChange.append(hard_disk)