from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def get_vmkernel_by_portgroup_new(self, port_group_name=None):
    """
        Check if vmkernel available or not
        Args:
            port_group_name: name of port group

        Returns: vmkernel managed object if vmkernel found, false if not

        """
    for vnic in self.esxi_host_obj.config.network.vnic:
        if vnic.spec.portgroup == port_group_name:
            return vnic
        try:
            if vnic.spec.distributedVirtualPort.portgroupKey == self.port_group_obj.key:
                return vnic
        except AttributeError:
            pass
    return False