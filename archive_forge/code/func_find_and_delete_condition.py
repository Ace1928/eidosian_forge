from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.waf import MATCH_LOOKUP
from ansible_collections.amazon.aws.plugins.module_utils.waf import get_rule_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_regional_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import list_rules_with_backoff
from ansible_collections.amazon.aws.plugins.module_utils.waf import run_func_with_change_token_backoff
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_and_delete_condition(self, condition_set_id):
    current_condition = self.get_condition_by_id(condition_set_id)
    in_use_rules = self.find_condition_in_rules(condition_set_id)
    if in_use_rules:
        rulenames = ', '.join(in_use_rules)
        self.module.fail_json(msg=f'Condition {current_condition['Name']} is in use by {rulenames}')
    if current_condition[self.conditiontuples]:
        func = getattr(self.client, 'update_' + self.method_suffix)
        params = self.format_for_deletion(current_condition)
        try:
            run_func_with_change_token_backoff(self.client, self.module, params, func)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Could not delete filters from condition')
    func = getattr(self.client, 'delete_' + self.method_suffix)
    params = dict()
    params[self.conditionsetid] = condition_set_id
    try:
        run_func_with_change_token_backoff(self.client, self.module, params, func, wait=True)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Could not delete condition')
    if self.type == 'regex':
        self.tidy_up_regex_patterns(current_condition)
    return (True, {})