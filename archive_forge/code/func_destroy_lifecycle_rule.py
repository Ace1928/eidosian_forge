import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def destroy_lifecycle_rule(client, module):
    name = module.params.get('name')
    prefix = module.params.get('prefix')
    rule_id = module.params.get('rule_id')
    wait = module.params.get('wait')
    changed = False
    if prefix is None:
        prefix = ''
    current_lifecycle_rules = fetch_rules(client, module, name)
    changed, lifecycle_obj = compare_and_remove_rule(current_lifecycle_rules, rule_id, prefix)
    if changed:
        try:
            if lifecycle_obj['Rules']:
                client.put_bucket_lifecycle_configuration(aws_retry=True, Bucket=name, LifecycleConfiguration=lifecycle_obj)
            elif current_lifecycle_rules:
                changed = True
                client.delete_bucket_lifecycle(aws_retry=True, Bucket=name)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e)
        _changed = changed
        _retries = 10
        _not_changed_cnt = 6
        while wait and _changed and _retries and _not_changed_cnt:
            time.sleep(5)
            _retries -= 1
            new_rules = fetch_rules(client, module, name)
            _changed, lifecycle_configuration = compare_and_remove_rule(new_rules, rule_id, prefix)
            if not _changed:
                _not_changed_cnt -= 1
                _changed = True
            else:
                _not_changed_cnt = 6
    else:
        _retries = 0
    new_rules = fetch_rules(client, module, name)
    module.exit_json(changed=changed, rules=new_rules, old_rules=current_lifecycle_rules, _retries=_retries)