import json
import time
from ansible.module_utils.basic import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.s3 import s3_extra_params
from ansible_collections.amazon.aws.plugins.module_utils.s3 import validate_bucket_name
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_bucket_ownership_cntrl(s3_client, bucket_name):
    """
    Get current bucket public access block
    """
    try:
        bucket_ownership = s3_client.get_bucket_ownership_controls(Bucket=bucket_name)
        return bucket_ownership['OwnershipControls']['Rules'][0]['ObjectOwnership']
    except is_boto3_error_code(['OwnershipControlsNotFoundError', 'NoSuchOwnershipControls']):
        return None