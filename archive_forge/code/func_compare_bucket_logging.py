from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_bucket_logging(bucket_logging, target_bucket, target_prefix):
    if not bucket_logging.get('LoggingEnabled', False):
        if target_bucket:
            return True
        return False
    logging = bucket_logging['LoggingEnabled']
    if logging['TargetBucket'] != target_bucket:
        return True
    if logging['TargetPrefix'] != target_prefix:
        return True
    return False