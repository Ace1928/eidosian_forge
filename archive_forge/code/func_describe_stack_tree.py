import datetime
import itertools
import time
import uuid
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_stack_tree(module, stack_set_name, operation_ids=None):
    jittered_backoff_decorator = AWSRetry.jittered_backoff(retries=5, delay=3, max_delay=5, catch_extra_error_codes=['StackSetNotFound'])
    cfn = module.client('cloudformation', retry_decorator=jittered_backoff_decorator)
    result = dict()
    result['stack_set'] = camel_dict_to_snake_dict(cfn.describe_stack_set(StackSetName=stack_set_name, aws_retry=True)['StackSet'])
    result['stack_set']['tags'] = boto3_tag_list_to_ansible_dict(result['stack_set']['tags'])
    result['operations_log'] = sorted(camel_dict_to_snake_dict(cfn.list_stack_set_operations(StackSetName=stack_set_name, aws_retry=True))['summaries'], key=lambda x: x['creation_timestamp'])
    result['stack_instances'] = sorted([camel_dict_to_snake_dict(i) for i in cfn.list_stack_instances(StackSetName=stack_set_name)['Summaries']], key=lambda i: i['region'] + i['account'])
    if operation_ids:
        result['operations'] = []
        for op_id in operation_ids:
            try:
                result['operations'].append(camel_dict_to_snake_dict(cfn.describe_stack_set_operation(StackSetName=stack_set_name, OperationId=op_id)['StackSetOperation']))
            except is_boto3_error_code('OperationNotFoundException'):
                pass
    return result