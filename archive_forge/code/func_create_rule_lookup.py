import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_rule_lookup(client, module):
    if client.__class__.__name__ == 'WAF':
        try:
            rules = list_rules_with_backoff(client)
            return dict(((rule['Name'], rule) for rule in rules))
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not list rules')
    elif client.__class__.__name__ == 'WAFRegional':
        try:
            rules = list_regional_rules_with_backoff(client)
            return dict(((rule['Name'], rule) for rule in rules))
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not list regional rules')