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
def put_bucket_encryption_with_retry(module, s3_client, name, expected_encryption):
    max_retries = 3
    for retries in range(1, max_retries + 1):
        try:
            put_bucket_encryption(s3_client, name, expected_encryption)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to set bucket encryption')
        current_encryption = wait_encryption_is_applied(module, s3_client, name, expected_encryption, should_fail=retries == max_retries, retries=5)
        if current_encryption == expected_encryption:
            return current_encryption
    module.fail_json(msg='Failed to apply bucket encryption', current=current_encryption, expected=expected_encryption, retries=retries)