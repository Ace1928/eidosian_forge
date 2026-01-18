import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.waf import MATCH_LOOKUP
from ansible_collections.amazon.aws.plugins.module_utils.waf import get_web_acl_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_web_acls_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_rule_absent(client, module):
    rule_id = get_rule_by_name(client, module, module.params['name'])
    in_use_web_acls = find_rule_in_web_acls(client, module, rule_id)
    if in_use_web_acls:
        web_acl_names = ', '.join(in_use_web_acls)
        module.fail_json(msg=f'Rule {module.params['name']} is in use by Web ACL(s) {web_acl_names}')
    if rule_id:
        remove_rule_conditions(client, module, rule_id)
        try:
            return (True, run_func_with_change_token_backoff(client, module, {'RuleId': rule_id}, client.delete_rule, wait=True))
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Could not delete rule')
    return (False, {})