from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def check_ctl_disk_exist(self, ctl_type=None, bus_number=None):
    """
        Check if controller of specified type exists and if there is disk attaching to it
        Return: Specified controller device, True or False of attaching disks
        """
    ctl_specified = None
    disks_attached_exist = False
    if ctl_type is None:
        return (ctl_specified, disks_attached_exist)
    for device in self.current_vm_obj.config.hardware.device:
        if isinstance(device, self.controller_types.get(ctl_type)):
            if bus_number is not None and device.busNumber != bus_number:
                continue
            ctl_specified = device
            if len(device.device) != 0:
                disks_attached_exist = True
            break
    return (ctl_specified, disks_attached_exist)