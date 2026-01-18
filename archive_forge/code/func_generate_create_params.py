import json
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import add_role_to_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import create_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import delete_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_instance_profiles
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_attached_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import remove_role_from_iam_instance_profile
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def generate_create_params(module):
    params = dict()
    params['Path'] = module.params.get('path') or '/'
    params['RoleName'] = module.params.get('name')
    params['AssumeRolePolicyDocument'] = module.params.get('assume_role_policy_document')
    if module.params.get('description') is not None:
        params['Description'] = module.params.get('description')
    if module.params.get('max_session_duration') is not None:
        params['MaxSessionDuration'] = module.params.get('max_session_duration')
    if module.params.get('boundary') is not None:
        params['PermissionsBoundary'] = module.params.get('boundary')
    if module.params.get('tags') is not None:
        params['Tags'] = ansible_dict_to_boto3_tag_list(module.params.get('tags'))
    return params