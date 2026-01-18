import copy
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def compare_delivery_policies(policy_a, policy_b):
    _policy_a = copy.deepcopy(policy_a)
    _policy_b = copy.deepcopy(policy_b)
    if 'http' in policy_a:
        if 'disableSubscriptionOverrides' not in policy_a['http']:
            _policy_a['http']['disableSubscriptionOverrides'] = False
    if 'http' in policy_b:
        if 'disableSubscriptionOverrides' not in policy_b['http']:
            _policy_b['http']['disableSubscriptionOverrides'] = False
    comparison = _policy_a != _policy_b
    return comparison