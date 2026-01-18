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
@IAMErrorHandler.deletion_error_handler('delete role')
def destroy_role(client, check_mode, role_name, delete_profiles):
    role = get_iam_role(client, role_name)
    if role is None:
        return False
    if check_mode:
        return True
    remove_instance_profiles(client, check_mode, role_name, delete_profiles)
    update_managed_policies(client, check_mode, role_name, [], True)
    remove_inline_policies(client, role_name)
    client.delete_role(aws_retry=True, RoleName=role_name)
    return True