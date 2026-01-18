from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def get_serial_port_info(vm_obj):
    """
    Get the serial port info
    """
    serial_port_info = []
    if vm_obj is None:
        return serial_port_info
    for port in vm_obj.config.hardware.device:
        backing = dict()
        if isinstance(port, vim.vm.device.VirtualSerialPort):
            if isinstance(port.backing, vim.vm.device.VirtualSerialPort.URIBackingInfo):
                backing['backing_type'] = 'network'
                backing['direction'] = port.backing.direction
                backing['service_uri'] = port.backing.serviceURI
                backing['proxy_uri'] = port.backing.proxyURI
            elif isinstance(port.backing, vim.vm.device.VirtualSerialPort.PipeBackingInfo):
                backing['backing_type'] = 'pipe'
                backing['pipe_name'] = port.backing.pipeName
                backing['endpoint'] = port.backing.endpoint
                backing['no_rx_loss'] = port.backing.noRxLoss
            elif isinstance(port.backing, vim.vm.device.VirtualSerialPort.DeviceBackingInfo):
                backing['backing_type'] = 'device'
                backing['device_name'] = port.backing.deviceName
            elif isinstance(port.backing, vim.vm.device.VirtualSerialPort.FileBackingInfo):
                backing['backing_type'] = 'file'
                backing['file_path'] = port.backing.fileName
            else:
                continue
            serial_port_info.append(backing)
    return serial_port_info