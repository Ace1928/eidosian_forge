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
@IAMErrorHandler.common_error_handler('set tags for user')
def ensure_user_tags(connection, check_mode, user, user_name, new_tags, purge_tags):
    if new_tags is None:
        return False
    existing_tags = user['tags']
    tags_to_add, tags_to_remove = compare_aws_tags(existing_tags, new_tags, purge_tags=purge_tags)
    if not tags_to_remove and (not tags_to_add):
        return False
    if check_mode:
        return True
    if tags_to_remove:
        connection.untag_user(UserName=user_name, TagKeys=tags_to_remove)
    if tags_to_add:
        connection.tag_user(UserName=user_name, Tags=ansible_dict_to_boto3_tag_list(tags_to_add))
    return True