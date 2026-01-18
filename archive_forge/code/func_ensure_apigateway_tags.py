import json
import traceback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_apigateway_tags(module, client, api_id, current_tags, new_tags, purge_tags):
    changed = False
    tag_result = {}
    tags_to_set, tags_to_delete = compare_aws_tags(current_tags, new_tags, purge_tags)
    if tags_to_set or tags_to_delete:
        changed = True
        apigateway_arn = f'arn:aws:apigateway:{module.region}::/restapis/{api_id}'
        if tags_to_delete:
            client.untag_resource(resourceArn=apigateway_arn, tagKeys=tags_to_delete)
        if tags_to_set:
            client.tag_resource(resourceArn=apigateway_arn, tags=tags_to_set)
        tag_result = get_rest_api(module, client, api_id=api_id)
    return (changed, tag_result)