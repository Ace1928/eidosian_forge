from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
@staticmethod
def create_lacp_group_spec(operation, key, name, uplink_number, mode, load_balancing_mode):
    """
            Create LACP group spec
            operation: add, edit, or remove
            Returns: LACP group spec
        """
    lacp_spec = vim.dvs.VmwareDistributedVirtualSwitch.LacpGroupSpec()
    lacp_spec.operation = operation
    lacp_spec.lacpGroupConfig = vim.dvs.VmwareDistributedVirtualSwitch.LacpGroupConfig()
    lacp_spec.lacpGroupConfig.name = name
    if operation in ('edit', 'remove'):
        lacp_spec.lacpGroupConfig.key = key
    if not operation == 'remove':
        lacp_spec.lacpGroupConfig.uplinkNum = uplink_number
        lacp_spec.lacpGroupConfig.mode = mode
        lacp_spec.lacpGroupConfig.loadbalanceAlgorithm = load_balancing_mode
    return lacp_spec