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
@IAMErrorHandler.list_error_handler('list MFA devices')
def delete_mfa_devices(connection, check_mode, user_name):
    devices = connection.list_mfa_devices(aws_retry=True, UserName=user_name)['MFADevices']
    if not devices:
        return False
    for device in devices:
        delete_mfa_device(connection, check_mode, user_name, device['SerialNumber'])
    return True