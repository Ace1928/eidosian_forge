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
@IAMErrorHandler.common_error_handler('add role to instance profile')
@AWSRetry.jittered_backoff()
def add_role_to_iam_instance_profile(client, profile_name, role_name):
    client.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
    return True