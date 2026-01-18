from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_unused_ctl_bus_number(self):
    """
        Get gid of occupied bus numbers of each type of disk controller, update the available bus number list
        """
    for device in self.current_vm_obj.config.hardware.device:
        if isinstance(device, self.device_helper.sata_device_type):
            if len(self.disk_ctl_bus_num_list['sata']) != 0:
                self.disk_ctl_bus_num_list['sata'].remove(device.busNumber)
        if isinstance(device, self.device_helper.nvme_device_type):
            if len(self.disk_ctl_bus_num_list['nvme']) != 0:
                self.disk_ctl_bus_num_list['nvme'].remove(device.busNumber)
        if isinstance(device, tuple(self.device_helper.scsi_device_type.values())):
            if len(self.disk_ctl_bus_num_list['scsi']) != 0:
                self.disk_ctl_bus_num_list['scsi'].remove(device.busNumber)