import re
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .arn import parse_aws_arn
from .arn import validate_aws_arn
from .botocore import is_boto3_error_code
from .botocore import normalize_boto3_result
from .errors import AWSErrorHandler
from .exceptions import AnsibleAWSError
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
@IAMErrorHandler.common_error_handler('create instance profile')
@AWSRetry.jittered_backoff()
def create_iam_instance_profile(client, name, path, tags):
    boto3_tags = ansible_dict_to_boto3_tag_list(tags or {})
    path = path or '/'
    result = client.create_instance_profile(InstanceProfileName=name, Path=path, Tags=boto3_tags)
    return result['InstanceProfile']