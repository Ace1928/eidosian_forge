from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_endpoint(connection, endpoint_identifier):
    """checks if the endpoint exists"""
    endpoint_filter = dict(Name='endpoint-id', Values=[endpoint_identifier])
    try:
        endpoints = dms_describe_endpoints(connection, Filters=[endpoint_filter])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe the DMS endpoint.')
    if not endpoints:
        return None
    endpoint = endpoints[0]
    try:
        tags = dms_describe_tags(connection, ResourceArn=endpoint['EndpointArn'])
        endpoint['tags'] = tags
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe the DMS endpoint tags')
    return endpoint