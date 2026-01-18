from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
def hostgroup_to_dict(self, hostgroup):
    result = hostgroup.as_dict()
    result['tags'] = hostgroup.tags
    return result