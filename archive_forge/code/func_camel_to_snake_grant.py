import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def camel_to_snake_grant(grant):
    """camel_to_snake_grant snakifies everything except the encryption context"""
    constraints = grant.get('Constraints', {})
    result = camel_dict_to_snake_dict(grant)
    if 'EncryptionContextEquals' in constraints:
        result['constraints']['encryption_context_equals'] = constraints['EncryptionContextEquals']
    if 'EncryptionContextSubset' in constraints:
        result['constraints']['encryption_context_subset'] = constraints['EncryptionContextSubset']
    return result