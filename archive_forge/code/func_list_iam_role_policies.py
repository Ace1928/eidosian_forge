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
@IAMErrorHandler.list_error_handler('list policies for role', [])
@AWSRetry.jittered_backoff()
def list_iam_role_policies(client, role_name):
    paginator = client.get_paginator('list_role_policies')
    return paginator.paginate(RoleName=role_name).build_full_result()['PolicyNames']