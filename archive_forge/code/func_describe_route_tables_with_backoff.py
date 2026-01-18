from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
@AWSRetry.jittered_backoff()
def describe_route_tables_with_backoff(connection, **params):
    try:
        paginator = connection.get_paginator('describe_route_tables')
        return paginator.paginate(**params).build_full_result()
    except is_boto3_error_code('InvalidRouteTableID.NotFound'):
        return None