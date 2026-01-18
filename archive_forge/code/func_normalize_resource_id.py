from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils.six import string_types
def normalize_resource_id(self, value, pattern):
    """
        Return a proper resource id string..

        :param resource_id: It could be a resource name, resource id or dict containing parts from the pattern.
        :param pattern: pattern of resource is, just like in Azure Swagger
        """
    value_dict = {}
    if isinstance(value, string_types):
        value_parts = value.split('/')
        if len(value_parts) == 1:
            value_dict['name'] = value
        else:
            pattern_parts = pattern.split('/')
            if len(value_parts) != len(pattern_parts):
                return None
            for i in range(len(value_parts)):
                if pattern_parts[i].startswith('{'):
                    value_dict[pattern_parts[i][1:-1]] = value_parts[i]
                elif value_parts[i].lower() != pattern_parts[i].lower():
                    return None
    elif isinstance(value, dict):
        value_dict = value
    else:
        return None
    if not value_dict.get('subscription_id'):
        value_dict['subscription_id'] = self.subscription_id
    if not value_dict.get('resource_group'):
        value_dict['resource_group'] = self.resource_group
    for k in value_dict:
        if not '{' + k + '}' in pattern:
            return None
    return pattern.format(**value_dict)