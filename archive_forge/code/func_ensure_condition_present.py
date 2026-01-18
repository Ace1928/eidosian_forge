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
def ensure_condition_present(self):
    name = self.module.params['name']
    condition_set_id = self.get_condition_by_name(name)
    if condition_set_id:
        return self.find_and_update_condition(condition_set_id)
    else:
        params = dict()
        params['Name'] = name
        func = getattr(self.client, 'create_' + self.method_suffix)
        try:
            condition = run_func_with_change_token_backoff(self.client, self.module, params, func)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Could not create condition')
        return self.find_and_update_condition(condition[self.conditionset][self.conditionsetid])