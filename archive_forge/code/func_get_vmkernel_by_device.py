from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def get_vmkernel_by_device(self, device_name):
    """
        Check if vmkernel available or not
        Args:
            device_name: name of vmkernel device

        Returns: vmkernel managed object if vmkernel found, false if not

        """
    for vnic in self.esxi_host_obj.config.network.vnic:
        if vnic.device == device_name:
            return vnic
    return None