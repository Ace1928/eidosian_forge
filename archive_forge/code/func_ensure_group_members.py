from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import convert_managed_policy_names_to_arns
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_group
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def ensure_group_members(connection, module, group_info, users, purge_users):
    if users is None:
        return False
    group_name = group_info['Group']['GroupName']
    current_group_members = [member['UserName'] for member in group_info['Users']]
    members_to_add = list(set(users) - set(current_group_members))
    members_to_remove = []
    if purge_users:
        members_to_remove = list(set(current_group_members) - set(users))
    if not members_to_add and (not members_to_remove):
        return False
    if module.check_mode:
        return True
    add_group_members(connection, module, group_name, members_to_add)
    remove_group_members(connection, module, group_name, members_to_remove)
    return True