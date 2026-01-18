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
def stack_operation(module, cfn, stack_name, operation, events_limit, op_token=None):
    """gets the status of a stack while it is created/updated/deleted"""
    existed = []
    while True:
        try:
            stack = get_stack_facts(module, cfn, stack_name, raise_errors=True)
            existed.append('yes')
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError):
            if 'yes' in existed or operation == 'DELETE':
                ret = get_stack_events(cfn, stack_name, events_limit, op_token)
                ret.update({'changed': True, 'output': 'Stack Deleted'})
                return ret
            else:
                return {'changed': True, 'failed': True, 'output': 'Stack Not Found', 'exception': traceback.format_exc()}
        ret = get_stack_events(cfn, stack_name, events_limit, op_token)
        if not stack:
            if 'yes' in existed or operation == 'DELETE':
                ret = get_stack_events(cfn, stack_name, events_limit, op_token)
                ret.update({'changed': True, 'output': 'Stack Deleted'})
                return ret
            else:
                ret.update({'changed': False, 'failed': True, 'output': 'Stack not found.'})
                return ret
        elif stack['StackStatus'].endswith('ROLLBACK_COMPLETE') and operation != 'CREATE_CHANGESET':
            ret.update({'changed': True, 'failed': True, 'output': f'Problem with {operation}. Rollback complete'})
            return ret
        elif stack['StackStatus'] == 'DELETE_COMPLETE' and operation == 'CREATE':
            ret.update({'changed': True, 'failed': True, 'output': 'Stack create failed. Delete complete.'})
            return ret
        elif stack['StackStatus'].endswith('_COMPLETE'):
            ret.update({'changed': True, 'output': f'Stack {operation} complete'})
            return ret
        elif stack['StackStatus'].endswith('_ROLLBACK_FAILED'):
            ret.update({'changed': True, 'failed': True, 'output': f'Stack {operation} rollback failed'})
            return ret
        elif stack['StackStatus'].endswith('_FAILED'):
            ret.update({'changed': True, 'failed': True, 'output': f'Stack {operation} failed'})
            return ret
        else:
            time.sleep(5)
    return {'failed': True, 'output': 'Failed for unknown reasons.'}