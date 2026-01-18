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
def destroy_user(connection, module):
    user_name = module.params.get('name')
    user = get_iam_user(connection, user_name)
    if not user:
        module.exit_json(changed=False)
    if module.check_mode:
        module.exit_json(changed=True)
    remove_login_profile(connection, module.check_mode, user_name, True, False)
    delete_access_keys(connection, module.check_mode, user_name)
    delete_ssh_public_keys(connection, module.check_mode, user_name)
    delete_service_credentials(connection, module.check_mode, user_name)
    delete_signing_certificates(connection, module.check_mode, user_name)
    delete_mfa_devices(connection, module.check_mode, user_name)
    detach_all_policies(connection, module.check_mode, user_name)
    delete_inline_policies(connection, module.check_mode, user_name)
    remove_from_all_groups(connection, module.check_mode, user_name)
    changed = delete_user(connection, module.check_mode, user_name)
    module.exit_json(changed=changed)