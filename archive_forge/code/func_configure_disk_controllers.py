from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def configure_disk_controllers(self):
    """
        Do disk controller management, add or remove

        Return: Operation result
        """
    if self.params['gather_disk_controller_facts']:
        results = {'changed': False, 'failed': False, 'disk_controller_data': self.gather_disk_controller_facts()}
        return results
    controller_config = self.sanitize_disk_controller_config()
    for disk_ctl_config in controller_config:
        if disk_ctl_config and disk_ctl_config['state'] == 'present':
            if disk_ctl_config['type'] in self.device_helper.usb_device_type.keys():
                usb_exists, has_disks_attached = self.check_ctl_disk_exist(disk_ctl_config['type'])
                if usb_exists:
                    self.module.warn("'%s' USB controller already exists, can not add more." % disk_ctl_config['type'])
                else:
                    disk_controller_new = self.create_controller(disk_ctl_config['type'], disk_ctl_config.get('bus_sharing'))
                    self.config_spec.deviceChange.append(disk_controller_new)
                    self.change_detected = True
            elif disk_ctl_config.get('controller_number') is not None:
                disk_controller_new = self.create_controller(disk_ctl_config['type'], disk_ctl_config.get('bus_sharing'), disk_ctl_config.get('controller_number'))
                self.config_spec.deviceChange.append(disk_controller_new)
                self.change_detected = True
            elif disk_ctl_config['type'] in self.device_helper.scsi_device_type.keys():
                self.module.warn("Already 4 SCSI controllers, can not add new '%s' controller." % disk_ctl_config['type'])
            else:
                self.module.warn("Already 4 '%s' controllers, can not add new one." % disk_ctl_config['type'])
        elif disk_ctl_config and disk_ctl_config['state'] == 'absent':
            existing_ctl, has_disks_attached = self.check_ctl_disk_exist(disk_ctl_config['type'], disk_ctl_config.get('controller_number'))
            if existing_ctl is not None:
                if not has_disks_attached:
                    ctl_spec = vim.vm.device.VirtualDeviceSpec()
                    ctl_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.remove
                    ctl_spec.device = existing_ctl
                    self.config_spec.deviceChange.append(ctl_spec)
                    self.change_detected = True
                else:
                    self.module.warn("Can not remove specified controller, type '%s', bus number '%s', there are disks attaching to it." % (disk_ctl_config['type'], disk_ctl_config.get('controller_number')))
            else:
                self.module.warn("Can not find specified controller to remove, type '%s', bus number '%s'." % (disk_ctl_config['type'], disk_ctl_config.get('controller_number')))
    try:
        task = self.current_vm_obj.ReconfigVM_Task(spec=self.config_spec)
        wait_for_task(task)
    except vim.fault.InvalidDeviceSpec as e:
        self.module.fail_json(msg='Failed to configure controller on given virtual machine due to invalid device spec : %s' % to_native(e.msg), details='Please check ESXi server logs for more details.')
    except vim.fault.RestrictedVersion as e:
        self.module.fail_json(msg='Failed to reconfigure virtual machine due to product versioning restrictions: %s' % to_native(e.msg))
    except TaskError as task_e:
        self.module.fail_json(msg=to_native(task_e))
    if task.info.state == 'error':
        results = {'changed': self.change_detected, 'failed': True, 'msg': task.info.error.msg}
    else:
        if self.change_detected:
            time.sleep(self.sleep_time)
        results = {'changed': self.change_detected, 'failed': False, 'disk_controller_data': self.gather_disk_controller_facts()}
    return results