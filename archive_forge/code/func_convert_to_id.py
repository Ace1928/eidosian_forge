from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def convert_to_id(rule, key):
    if rule.get(key):
        ids = []
        for p in rule.get(key):
            if isinstance(p, dict):
                ids.append('/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/applicationSecurityGroups/{2}'.format(self.subscription_id, p.get('resource_group'), p.get('name')))
            elif isinstance(p, str):
                if is_valid_resource_id(p):
                    ids.append(p)
                else:
                    ids.append('/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Network/applicationSecurityGroups/{2}'.format(self.subscription_id, self.resource_group, p))
        rule[key] = ids