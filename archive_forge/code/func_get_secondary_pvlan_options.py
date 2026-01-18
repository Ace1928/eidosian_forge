from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_secondary_pvlan_options(self, secondary_pvlan):
    """Get Secondary PVLAN option"""
    secondary_pvlan_id = secondary_pvlan.get('secondary_pvlan_id', None)
    if secondary_pvlan_id is None:
        self.module.fail_json(msg="Please specify secondary_pvlan_id in secondary_pvlans options as it's a required parameter")
    primary_pvlan_id = secondary_pvlan.get('primary_pvlan_id', None)
    if primary_pvlan_id is None:
        self.module.fail_json(msg="Please specify primary_pvlan_id in secondary_pvlans options as it's a required parameter")
    if secondary_pvlan_id in (0, 4095) or primary_pvlan_id in (0, 4095):
        self.module.fail_json(msg='The VLAN IDs of 0 and 4095 are reserved and cannot be used as a primary or secondary PVLAN.')
    pvlan_type = secondary_pvlan.get('pvlan_type', None)
    supported_pvlan_types = ['isolated', 'community']
    if pvlan_type is None:
        self.module.fail_json(msg="Please specify pvlan_type in secondary_pvlans options as it's a required parameter")
    elif pvlan_type not in supported_pvlan_types:
        self.module.fail_json(msg="The specified PVLAN type '%s' isn't supported!" % pvlan_type)
    return (secondary_pvlan_id, primary_pvlan_id, pvlan_type)