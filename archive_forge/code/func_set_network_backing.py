from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def set_network_backing(self, serial_port, backing_info):
    """
        Set the networking backing params
        """
    required_params = ['service_uri', 'direction']
    if set(required_params).issubset(backing_info.keys()):
        backing = serial_port.URIBackingInfo()
        backing.serviceURI = backing_info['service_uri']
        backing.proxyURI = backing_info['proxy_uri']
        backing.direction = backing_info['direction']
    else:
        self.module.fail_json(msg='Failed to create a new serial port of network backing type due to insufficient parameters.' + 'The required parameters are service_uri and direction')
    return backing