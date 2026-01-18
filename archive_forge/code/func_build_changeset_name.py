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
def build_changeset_name(stack_params):
    if 'ChangeSetName' in stack_params:
        return stack_params['ChangeSetName']
    json_params = json.dumps(stack_params, sort_keys=True)
    changeset_sha = sha1(to_bytes(json_params, errors='surrogate_or_strict')).hexdigest()
    return f'Ansible-{stack_params['StackName']}-{changeset_sha}'