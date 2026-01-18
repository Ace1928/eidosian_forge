from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def diff_serial_port_config(serial_port, backing):
    if backing['state'] == 'present':
        if 'yield_on_poll' in backing:
            if serial_port.yieldOnPoll != backing['yield_on_poll']:
                return True
        if backing['service_uri'] is not None:
            if serial_port.backing.serviceURI != backing['service_uri'] or serial_port.backing.direction != backing['direction'] or serial_port.backing.proxyURI != backing['proxy_uri']:
                return True
        if backing['pipe_name'] is not None:
            if serial_port.backing.pipeName != backing['pipe_name'] or serial_port.backing.endpoint != backing['endpoint'] or serial_port.backing.noRxLoss != backing['no_rx_loss']:
                return True
        if backing['device_name'] is not None:
            if serial_port.backing.deviceName != backing['device_name']:
                return True
        if backing['file_path'] is not None:
            if serial_port.backing.fileName != backing['file_path']:
                return True
    if backing['state'] == 'absent':
        if backing['service_uri'] is not None:
            if serial_port.backing.serviceURI == backing['service_uri'] and serial_port.backing.proxyURI == backing['proxy_uri']:
                return True
        if backing['pipe_name'] is not None:
            if serial_port.backing.pipeName == backing['pipe_name']:
                return True
        if backing['device_name'] is not None:
            if serial_port.backing.deviceName == backing['device_name']:
                return True
        if backing['file_path'] is not None:
            if serial_port.backing.fileName == backing['file_path']:
                return True
    return False