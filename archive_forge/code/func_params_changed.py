from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def params_changed(state_machine_arn, sfn_client, module):
    """
    Check whether the state machine definition or IAM Role ARN is different
    from the existing state machine parameters.
    """
    current = sfn_client.describe_state_machine(stateMachineArn=state_machine_arn)
    return current.get('definition') != module.params.get('definition') or current.get('roleArn') != module.params.get('role_arn')