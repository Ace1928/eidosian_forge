import json
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import IAMErrorHandler
from ansible_collections.amazon.aws.plugins.module_utils.iam import detach_iam_group_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import detach_iam_role_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import detach_iam_user_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_managed_policy_by_arn
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_managed_policy_by_name
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_managed_policy_version
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_entities_for_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_managed_policy_versions
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import tag_iam_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import untag_iam_policy
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def detach_all_entities(policy):
    arn = policy['Arn']
    entities = list_iam_entities_for_policy(client, arn)
    if not entities:
        return False
    for g in entities['PolicyGroups']:
        detach_iam_group_policy(client, arn, g['GroupName'])
    for u in entities['PolicyUsers']:
        detach_iam_user_policy(client, arn, u['UserName'])
    for r in entities['PolicyRoles']:
        detach_iam_role_policy(client, arn, r['RoleName'])
    return True