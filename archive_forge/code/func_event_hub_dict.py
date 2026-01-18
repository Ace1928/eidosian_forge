from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def event_hub_dict(self, setting_dict):
    auth_rule_id = setting_dict.get('event_hub_authorization_rule_id')
    if auth_rule_id:
        parsed_rule_id = parse_resource_id(auth_rule_id)
        return dict(id=resource_id(subscription=parsed_rule_id.get('subscription'), resource_group=parsed_rule_id.get('resource_group'), namespace=parsed_rule_id.get('namespace'), type=parsed_rule_id.get('type'), name=parsed_rule_id.get('name')), namespace=parsed_rule_id.get('name'), hub=setting_dict.get('event_hub_name'), policy=parsed_rule_id.get('resource_name'))
    return None