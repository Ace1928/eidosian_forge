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
@IAMErrorHandler.common_error_handler('create policy')
def create_managed_policy(name, path, policy, description, tags):
    if module.check_mode:
        module.exit_json(changed=True)
    if policy is None:
        raise AnsibleIAMError(message='Managed policy would be created but policy parameter is missing')
    params = {'PolicyName': name, 'PolicyDocument': policy}
    if path:
        params['Path'] = path
    if description:
        params['Description'] = description
    if tags:
        params['Tags'] = ansible_dict_to_boto3_tag_list(tags)
    rvalue = client.create_policy(aws_retry=True, **params)
    new_policy = get_iam_managed_policy_by_arn(client, rvalue['Policy']['Arn'])
    module.exit_json(changed=True, policy=normalize_iam_policy(new_policy))