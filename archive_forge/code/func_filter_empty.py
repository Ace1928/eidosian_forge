from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def filter_empty(**kwargs):
    retval = {}
    for k, v in kwargs.items():
        if v:
            retval[k] = v
    return retval