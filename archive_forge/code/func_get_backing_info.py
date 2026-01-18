from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def get_backing_info(self, serial_port, backing, backing_type):
    """
        Returns the call to the appropriate backing function based on the backing type
        """
    switcher = {'network': self.set_network_backing, 'pipe': self.set_pipe_backing, 'device': self.set_device_backing, 'file': self.set_file_backing}
    backing_func = switcher.get(backing_type, None)
    if backing_func is None:
        self.module.fail_json(msg="Failed to find a valid backing type. Provided '%s', should be one of [%s]" % (backing_type, ', '.join(switcher.keys())))
    return backing_func(serial_port, backing)