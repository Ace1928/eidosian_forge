from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_user
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_user
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def ensure_permissions_boundary(connection, check_mode, user, user_name, boundary):
    if boundary is None:
        return False
    current_boundary = user.get('permissions_boundary', '') if user else None
    if current_boundary:
        current_boundary = current_boundary.get('permissions_boundary_arn')
    if boundary == current_boundary:
        return False
    if check_mode:
        return True
    if boundary == '':
        _delete_user_permissions_boundary(connection, check_mode, user_name)
    else:
        _put_user_permissions_boundary(connection, check_mode, user_name, boundary)
    return True