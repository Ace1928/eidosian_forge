from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def describe_wafv2_tags(wafv2, arn, fail_json_aws):
    next_marker = None
    tag_list = []
    while True:
        responce = _list_tags(wafv2, arn, fail_json_aws)
        next_marker = responce.get('NextMarker', None)
        tag_info = responce.get('TagInfoForResource', {})
        tag_list.extend(tag_info.get('TagList', []))
        if not next_marker:
            break
    return boto3_tag_list_to_ansible_dict(tag_list)