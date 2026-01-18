import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_rule(new_rule, old_rule, purge_transitions):
    rule1 = deepcopy(new_rule)
    rule2 = deepcopy(old_rule)
    if purge_transitions:
        return rule1 == rule2
    else:
        transitions1 = rule1.pop('Transitions', [])
        transitions2 = rule2.pop('Transitions', [])
        noncurrent_transtions1 = rule1.pop('NoncurrentVersionTransitions', [])
        noncurrent_transtions2 = rule2.pop('NoncurrentVersionTransitions', [])
        if rule1 != rule2:
            return False
        for transition in transitions1:
            if transition not in transitions2:
                return False
        for transition in noncurrent_transtions1:
            if transition not in noncurrent_transtions2:
                return False
        return True