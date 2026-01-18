import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_and_remove_rule(current_lifecycle_rules, rule_id=None, prefix=None):
    changed = False
    lifecycle_configuration = dict(Rules=[])
    if rule_id is not None:
        for existing_rule in current_lifecycle_rules:
            if rule_id == existing_rule['ID']:
                changed = True
            else:
                lifecycle_configuration['Rules'].append(existing_rule)
    else:
        for existing_rule in current_lifecycle_rules:
            if prefix == existing_rule['Filter'].get('Prefix', ''):
                changed = True
            else:
                lifecycle_configuration['Rules'].append(existing_rule)
    return (changed, lifecycle_configuration)