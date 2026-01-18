from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_host_vnic_config(self, dv_switch_uuid, portgroup_key):
    host_vnic_config = vim.host.VirtualNic.Config()
    host_vnic_config.spec = vim.host.VirtualNic.Specification()
    host_vnic_config.changeOperation = 'edit'
    host_vnic_config.device = self.device
    host_vnic_config.portgroup = ''
    host_vnic_config.spec.distributedVirtualPort = vim.dvs.PortConnection()
    host_vnic_config.spec.distributedVirtualPort.switchUuid = dv_switch_uuid
    host_vnic_config.spec.distributedVirtualPort.portgroupKey = portgroup_key
    return host_vnic_config