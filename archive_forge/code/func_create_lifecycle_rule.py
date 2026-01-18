import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_lifecycle_rule(client, module):
    name = module.params.get('name')
    wait = module.params.get('wait')
    changed = False
    old_lifecycle_rules = fetch_rules(client, module, name)
    new_rule = build_rule(client, module)
    changed, lifecycle_configuration = compare_and_update_configuration(client, module, old_lifecycle_rules, new_rule)
    if changed:
        try:
            client.put_bucket_lifecycle_configuration(aws_retry=True, Bucket=name, LifecycleConfiguration=lifecycle_configuration)
        except is_boto3_error_message('At least one action needs to be specified in a rule'):
            changed = False
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, lifecycle_configuration=lifecycle_configuration, name=name, old_lifecycle_rules=old_lifecycle_rules)
        _changed = changed
        _retries = 10
        _not_changed_cnt = 6
        while wait and _changed and _retries and _not_changed_cnt:
            time.sleep(5)
            _retries -= 1
            new_rules = fetch_rules(client, module, name)
            _changed, lifecycle_configuration = compare_and_update_configuration(client, module, new_rules, new_rule)
            if not _changed:
                _not_changed_cnt -= 1
                _changed = True
            else:
                _not_changed_cnt = 6
    else:
        _retries = 0
    new_rules = fetch_rules(client, module, name)
    module.exit_json(changed=changed, new_rule=new_rule, rules=new_rules, old_rules=old_lifecycle_rules, _retries=_retries, _config=lifecycle_configuration)