from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def get_serial_port(vm_obj, backing):
    """
    Return the serial port of specified backing type
    """
    serial_port = None
    backing_type_mapping = {'network': vim.vm.device.VirtualSerialPort.URIBackingInfo, 'pipe': vim.vm.device.VirtualSerialPort.PipeBackingInfo, 'device': vim.vm.device.VirtualSerialPort.DeviceBackingInfo, 'file': vim.vm.device.VirtualSerialPort.FileBackingInfo}
    valid_params = backing.keys()
    for device in vm_obj.config.hardware.device:
        if isinstance(device, vim.vm.device.VirtualSerialPort):
            backing_type = backing.get('type', backing.get('backing_type', None))
            if isinstance(device.backing, backing_type_mapping[backing_type]):
                if 'service_uri' in valid_params:
                    serial_port = device
                    break
                if 'pipe_name' in valid_params:
                    serial_port = device
                    break
                if 'device_name' in valid_params:
                    serial_port = device
                    break
                if 'file_path' in valid_params:
                    serial_port = device
                    break
                serial_port = device
    return serial_port