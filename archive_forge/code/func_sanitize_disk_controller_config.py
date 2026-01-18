from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def sanitize_disk_controller_config(self):
    """
        Check correctness of controller configuration provided by user

        Return: A list of dictionary with checked controller configured
        """
    if not self.params.get('controllers'):
        self.module.exit_json(changed=False, msg="No controller provided for virtual machine '%s' for management." % self.current_vm_obj.name)
    if 10 != self.params.get('sleep_time') <= 300:
        self.sleep_time = self.params.get('sleep_time')
    exec_get_unused_ctl_bus_number = False
    controller_config = self.params.get('controllers')
    for ctl_config in controller_config:
        if ctl_config:
            if ctl_config['type'] not in self.device_helper.usb_device_type.keys():
                if ctl_config['state'] == 'absent' and ctl_config.get('controller_number') is None:
                    self.module.fail_json(msg='Disk controller number is required when removing it.')
                if ctl_config['state'] == 'present' and (not exec_get_unused_ctl_bus_number):
                    self.get_unused_ctl_bus_number()
                    exec_get_unused_ctl_bus_number = True
            if ctl_config['state'] == 'present' and ctl_config['type'] == 'nvme':
                vm_hwv = int(self.current_vm_obj.config.version.split('-')[1])
                if vm_hwv < 13:
                    self.module.fail_json(msg="Can not create new NVMe disk controller due to VM hardware version is '%s', not >= 13." % vm_hwv)
    if exec_get_unused_ctl_bus_number:
        for ctl_config in controller_config:
            if ctl_config and ctl_config['state'] == 'present' and (ctl_config['type'] not in self.device_helper.usb_device_type.keys()):
                if ctl_config['type'] in self.device_helper.scsi_device_type.keys():
                    if len(self.disk_ctl_bus_num_list['scsi']) != 0:
                        ctl_config['controller_number'] = self.disk_ctl_bus_num_list['scsi'].pop(0)
                    else:
                        ctl_config['controller_number'] = None
                elif ctl_config['type'] == 'sata' or ctl_config['type'] == 'nvme':
                    if len(self.disk_ctl_bus_num_list.get(ctl_config['type'])) != 0:
                        ctl_config['controller_number'] = self.disk_ctl_bus_num_list.get(ctl_config['type']).pop(0)
                    else:
                        ctl_config['controller_number'] = None
    return controller_config