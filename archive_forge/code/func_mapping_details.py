import json
import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def mapping_details(client, module, function_name):
    """
    Returns all lambda event source mappings.

    :param client: AWS API client reference (boto3)
    :param module: Ansible module reference
    :param function_name (str): Name of Lambda function to query
    :return dict:
    """
    lambda_info = dict()
    params = dict()
    params['FunctionName'] = function_name
    if module.params.get('event_source_arn'):
        params['EventSourceArn'] = module.params.get('event_source_arn')
    try:
        lambda_info.update(mappings=_paginate(client, 'list_event_source_mappings', **params)['EventSourceMappings'])
    except is_boto3_error_code('ResourceNotFoundException'):
        lambda_info.update(mappings=[])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Trying to get source event mappings')
    return camel_dict_to_snake_dict(lambda_info)