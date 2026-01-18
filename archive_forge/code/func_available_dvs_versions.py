from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def available_dvs_versions(self):
    """Get the DVS version supported by the vCenter"""
    dvs_mng = self.content.dvSwitchManager
    available_dvs_specs = dvs_mng.QueryAvailableDvsSpec(recommended=True)
    available_dvs_versions = []
    for available_dvs_spec in available_dvs_specs:
        available_dvs_versions.append(available_dvs_spec.version)
    return available_dvs_versions