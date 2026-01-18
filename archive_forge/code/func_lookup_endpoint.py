from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def lookup_endpoint(self, target, routing_endpoints):
    resource_type = target['resource_type']
    attribute = routing_endpoints_resource_type_mapping[resource_type]['attribute']
    endpoints = getattr(routing_endpoints, attribute)
    if not endpoints or len(endpoints) == 0:
        return False
    for item in endpoints:
        if item.name == target['name']:
            if target.get('resource_group') and target['resource_group'] != (item.resource_group or self.resource_group):
                return False
            if target.get('subscription_id') and target['subscription_id'] != (item.subscription_id or self.subscription_id):
                return False
            connection_string_regex = item.connection_string.replace('****', '.*')
            connection_string_regex = re.sub(':\\d+/;', '/;', connection_string_regex)
            if not re.search(connection_string_regex, target['connection_string']):
                return False
            if resource_type == 'storage':
                if target.get('container') and item.container_name != target['container']:
                    return False
                if target.get('encoding') and item.encoding != target['encoding']:
                    return False
            return True
    return False