from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def parse_nsg(self):
    nsg = self.security_group
    resource_group = self.resource_group
    if isinstance(self.security_group, dict):
        nsg = self.security_group.get('name')
        resource_group = self.security_group.get('resource_group', self.resource_group)
    id = format_resource_id(val=nsg, subscription_id=self.subscription_id, namespace='Microsoft.Network', types='networkSecurityGroups', resource_group=resource_group)
    name = azure_id_to_dict(id).get('name')
    return dict(id=id, name=name)