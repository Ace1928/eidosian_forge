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
def create_or_update_policy(existing_policy):
    name = module.params.get('name')
    path = module.params.get('path')
    description = module.params.get('description')
    default = module.params.get('make_default')
    only = module.params.get('only_version')
    tags = module.params.get('tags')
    purge_tags = module.params.get('purge_tags')
    policy = None
    if module.params.get('policy') is not None:
        policy = json.dumps(json.loads(module.params.get('policy')))
    if existing_policy is None:
        create_managed_policy(name, path, policy, description, tags)
    else:
        update_managed_policy(existing_policy, path, policy, description, default, only, tags, purge_tags)