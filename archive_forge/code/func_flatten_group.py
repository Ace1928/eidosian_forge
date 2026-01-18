from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def flatten_group(self, management_group):
    management_group_list = []
    subscription_list = []
    if management_group.get('children'):
        for child in management_group.get('children', []):
            if child.get('type') == '/providers/Microsoft.Management/managementGroups':
                management_group_list.append(child)
                new_groups, new_subscriptions = self.flatten_group(child)
                management_group_list += new_groups
                subscription_list += new_subscriptions
            elif child.get('type') == '/subscriptions':
                subscription_list.append(child)
    return (management_group_list, subscription_list)