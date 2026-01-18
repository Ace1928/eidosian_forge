import json
import time
import traceback
import uuid
from hashlib import sha1
from ansible.module_utils._text import to_bytes
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.botocore import boto_exception
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def get_stack_facts(module, cfn, stack_name, raise_errors=False):
    try:
        stack_response = cfn.describe_stacks(aws_retry=True, StackName=stack_name)
        stack_info = stack_response['Stacks'][0]
    except is_boto3_error_message('does not exist'):
        return None
    except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as err:
        if raise_errors:
            raise err
        module.fail_json_aws(err, msg='Failed to describe stack')
    if stack_response and stack_response.get('Stacks', None):
        stacks = stack_response['Stacks']
        if len(stacks):
            stack_info = stacks[0]
    return stack_info