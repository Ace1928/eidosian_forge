import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff()
def describe_parameter(client, module, **args):
    paginator = client.get_paginator('describe_parameters')
    existing_parameter = paginator.paginate(**args).build_full_result()
    if not existing_parameter['Parameters']:
        return None
    tags_dict = get_parameter_tags(client, module, module.params.get('name'))
    existing_parameter['Parameters'][0]['tags'] = tags_dict
    return existing_parameter['Parameters'][0]