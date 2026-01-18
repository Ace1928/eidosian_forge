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
def find_and_update_condition(self, condition_set_id):
    current_condition = self.get_condition_by_id(condition_set_id)
    update = self.format_for_update(condition_set_id)
    missing = self.find_missing(update, current_condition)
    if self.module.params.get('purge_filters'):
        extra = [{'Action': 'DELETE', self.conditiontuple: current_tuple} for current_tuple in current_condition[self.conditiontuples] if current_tuple not in [desired[self.conditiontuple] for desired in update['Updates']]]
    else:
        extra = []
    changed = bool(missing or extra)
    if changed:
        update['Updates'] = missing + extra
        func = getattr(self.client, 'update_' + self.method_suffix)
        try:
            result = run_func_with_change_token_backoff(self.client, self.module, update, func, wait=True)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Could not update condition')
    return (changed, self.get_condition_by_id(condition_set_id))