from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def set_device_backing(self, serial_port, backing_info):
    """
        Set the device backing params
        """
    required_params = ['device_name']
    if set(required_params).issubset(backing_info.keys()):
        backing = serial_port.DeviceBackingInfo()
        backing.deviceName = backing_info['device_name']
    else:
        self.module.fail_json(msg='Failed to create a new serial port of device backing type due to insufficient parameters.' + 'The required parameters are device_name')
    return backing