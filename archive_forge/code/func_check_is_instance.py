from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def check_is_instance(device_id, in_vpc):
    if not device_id:
        return False
    if device_id.startswith('i-'):
        return True
    if device_id.startswith('eni-') and (not in_vpc):
        raise EipError('If you are specifying an ENI, in_vpc must be true')
    return False