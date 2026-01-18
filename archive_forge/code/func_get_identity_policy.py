import json
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_identity_policy(connection, module, identity, policy_name):
    try:
        response = connection.get_identity_policies(Identity=identity, PolicyNames=[policy_name], aws_retry=True)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed to retrieve identity policy {policy_name}')
    policies = response['Policies']
    if policy_name in policies:
        return policies[policy_name]
    return None