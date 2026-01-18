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
def check_mode_changeset(module, stack_params, cfn):
    """Create a change set, describe it and delete it before returning check mode outputs."""
    stack_params['ChangeSetName'] = build_changeset_name(stack_params)
    stack_params.pop('ClientRequestToken', None)
    try:
        change_set = cfn.create_change_set(aws_retry=True, **stack_params)
        for _i in range(60):
            description = cfn.describe_change_set(aws_retry=True, ChangeSetName=change_set['Id'])
            if description['Status'] in ('CREATE_COMPLETE', 'FAILED'):
                break
            time.sleep(5)
        else:
            module.fail_json(msg=f'Failed to create change set {stack_params['ChangeSetName']}')
        cfn.delete_change_set(aws_retry=True, ChangeSetName=change_set['Id'])
        reason = description.get('StatusReason')
        if description['Status'] == 'FAILED' and ("didn't contain changes" in reason or 'No updates are to be performed' in reason):
            return {'changed': False, 'msg': reason, 'meta': reason}
        return {'changed': True, 'msg': reason, 'meta': description['Changes']}
    except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as err:
        module.fail_json_aws(err)