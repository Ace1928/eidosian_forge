from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def get_key_provider_type(self, kmip_cluster_info):
    key_provider_type = ''
    if kmip_cluster_info is None:
        return key_provider_type
    if not self.vcenter_version_at_least(version=(7, 0, 2)):
        key_provider_type = 'standard'
    elif kmip_cluster_info.managementType == 'vCenter':
        key_provider_type = 'standard'
    elif kmip_cluster_info.managementType == 'nativeProvider':
        key_provider_type = 'native'
    else:
        key_provider_type = kmip_cluster_info.managementType
    return key_provider_type