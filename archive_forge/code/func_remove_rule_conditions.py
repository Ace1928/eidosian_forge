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
def remove_rule_conditions(client, module, rule_id):
    conditions = get_rule(client, module, rule_id)['Predicates']
    updates = [format_for_deletion(camel_dict_to_snake_dict(condition)) for condition in conditions]
    try:
        run_func_with_change_token_backoff(client, module, {'RuleId': rule_id, 'Updates': updates}, client.update_rule)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Could not remove rule conditions')