from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def reconfigure_vm_serial_port(self, vm_obj):
    """
        Reconfigure vm with new or modified serial port config spec
        """
    self.get_serial_port_config_spec(vm_obj)
    try:
        if self.serial_ports:
            for serial_port in self.serial_ports:
                spec = vim.vm.ConfigSpec()
                spec.deviceChange.append(serial_port)
                task = vm_obj.ReconfigVM_Task(spec=spec)
                wait_for_task(task)
        task = vm_obj.ReconfigVM_Task(spec=self.config_spec)
        wait_for_task(task)
    except vim.fault.InvalidDatastorePath as e:
        self.module.fail_json(msg='Failed to configure serial port on given virtual machine due to invalid path: %s' % to_native(e.msg))
    except vim.fault.RestrictedVersion as e:
        self.module.fail_json(msg='Failed to reconfigure virtual machine due to product versioning restrictions: %s' % to_native(e.msg))
    if task.info.state == 'error':
        results = {'changed': self.change_applied, 'failed': True, 'msg': task.info.error.msg}
    else:
        serial_port_info = get_serial_port_info(vm_obj)
        results = {'changed': self.change_applied, 'failed': False, 'serial_port_info': serial_port_info}
    return results