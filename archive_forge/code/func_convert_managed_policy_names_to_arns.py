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
def convert_managed_policy_names_to_arns(client, policy_names):
    if all((validate_aws_arn(policy, service='iam') for policy in policy_names if policy is not None)):
        return policy_names
    allpolicies = {}
    policies = list_iam_managed_policies(client)
    for policy in policies:
        allpolicies[policy['PolicyName']] = policy['Arn']
        allpolicies[policy['Arn']] = policy['Arn']
    try:
        return [allpolicies[policy] for policy in policy_names if policy is not None]
    except KeyError as e:
        raise AnsibleIAMError(message='Failed to find policy by name:' + str(e), exception=e) from e