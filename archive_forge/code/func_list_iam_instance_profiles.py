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
@IAMErrorHandler.list_error_handler('list instance profiles', [])
def list_iam_instance_profiles(client, name=None, prefix=None, role=None):
    """
    Returns a list of IAM instance profiles in boto3 format.
    Profiles need to be converted to Ansible format using normalize_iam_instance_profile before being displayed.

    See also: normalize_iam_instance_profile
    """
    if role:
        return _list_iam_instance_profiles_for_role(client, RoleName=role)
    if name:
        return [_get_iam_instance_profiles(client, InstanceProfileName=name)]
    if prefix:
        return _list_iam_instance_profiles(client, PathPrefix=prefix)
    return _list_iam_instance_profiles(client)