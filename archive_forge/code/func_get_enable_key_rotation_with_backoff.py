import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
@AWSRetry.jittered_backoff(retries=5, delay=5, backoff=2.0)
def get_enable_key_rotation_with_backoff(connection, key_id):
    try:
        current_rotation_status = connection.get_key_rotation_status(KeyId=key_id)
    except is_boto3_error_code(['AccessDeniedException', 'UnsupportedOperationException']):
        return None
    return current_rotation_status.get('KeyRotationEnabled')