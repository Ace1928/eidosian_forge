from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_state_machine_arn(sfn_client, module):
    """
    Finds the state machine ARN based on the name parameter. Returns None if
    there is no state machine with this name.
    """
    target_name = module.params.get('name')
    all_state_machines = sfn_client.list_state_machines(aws_retry=True).get('stateMachines')
    for state_machine in all_state_machines:
        if state_machine.get('name') == target_name:
            return state_machine.get('stateMachineArn')