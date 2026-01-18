from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import compare_priority_rules
from ansible_collections.community.aws.plugins.module_utils.wafv2 import describe_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import ensure_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_list_rule_groups
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_snake_dict_to_camel_dict
def refresh_group(self):
    existing_group = None
    if self.id:
        try:
            response = self.wafv2.get_rule_group(Name=self.name, Scope=self.scope, Id=self.id)
            existing_group = response.get('RuleGroup')
            self.locktoken = response.get('LockToken')
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to get wafv2 rule group.')
        tags = describe_wafv2_tags(self.wafv2, self.arn, self.fail_json_aws)
        existing_group['tags'] = tags or {}
    return existing_group