from ansible_collections.amazon.aws.plugins.module_utils.iam import AnsibleIAMError
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_instance_profiles
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_attached_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_role_policies
from ansible_collections.amazon.aws.plugins.module_utils.iam import list_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.iam import normalize_iam_role
from ansible_collections.amazon.aws.plugins.module_utils.iam import validate_iam_identifiers
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def describe_iam_roles(client, name, path_prefix):
    if name:
        roles = [get_iam_role(client, name)]
    else:
        roles = list_iam_roles(client, path=path_prefix)
    roles = [r for r in roles if r is not None]
    return [normalize_iam_role(expand_iam_role(client, role), _v7_compat=True) for role in roles]