from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def find_dvspg_by_key(self, dv_switch, portgroup_key):
    """
        Find dvPortgroup by key
        Returns: dvPortgroup name
        """
    portgroups = dv_switch.portgroup
    for portgroup in portgroups:
        if portgroup.key == portgroup_key:
            return portgroup.name
    return None