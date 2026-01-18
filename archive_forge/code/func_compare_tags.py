from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_tags(state_machine_arn, sfn_client, module):
    new_tags = module.params.get('tags')
    current_tags = sfn_client.list_tags_for_resource(resourceArn=state_machine_arn).get('tags')
    return compare_aws_tags(boto3_tag_list_to_ansible_dict(current_tags), new_tags if new_tags else {}, module.params.get('purge_tags'))