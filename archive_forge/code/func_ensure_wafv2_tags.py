from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def ensure_wafv2_tags(wafv2, arn, tags, purge_tags, fail_json_aws, check_mode):
    if tags is None:
        return False
    current_tags = describe_wafv2_tags(wafv2, arn, fail_json_aws)
    tags_to_add, tags_to_remove = compare_aws_tags(current_tags, tags, purge_tags)
    if not tags_to_add and (not tags_to_remove):
        return False
    if check_mode:
        return True
    if tags_to_add:
        try:
            boto3_tags = ansible_dict_to_boto3_tag_list(tags_to_add)
            wafv2.tag_resource(ResourceARN=arn, Tags=boto3_tags)
        except (BotoCoreError, ClientError) as e:
            fail_json_aws(e, msg='Failed to add wafv2 tags')
    if tags_to_remove:
        try:
            wafv2.untag_resource(ResourceARN=arn, TagKeys=tags_to_remove)
        except (BotoCoreError, ClientError) as e:
            fail_json_aws(e, msg='Failed to remove wafv2 tags')
    return True