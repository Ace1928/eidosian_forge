import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def stop_stream_encryption(client, stream_name, encryption_type='', key_id='', wait=True, wait_timeout=300, check_mode=False):
    """Stop encryption on an Amazon Kinesis Stream.
    Args:
        client (botocore.client.EC2): Boto3 client.
        stream_name (str): The name of the kinesis stream.

    Kwargs:
        encryption_type (str): KMS or NONE
        key_id (str): KMS key GUID or alias
        wait (bool): Wait until Stream is ACTIVE.
            default=False
        wait_timeout (int): How long to wait until this operation is considered failed.
            default=300
        check_mode (bool): This will pass DryRun as one of the parameters to the aws api.
            default=False

    Basic Usage:
        >>> client = boto3.client('kinesis')
        >>> stream_name = 'test-stream'
        >>> stop_stream_encryption(client, stream_name,encryption_type, key_id)

    Returns:
        Tuple (bool, bool, str, dict)
    """
    success = False
    changed = False
    err_msg = ''
    params = {'StreamName': stream_name}
    results = dict()
    stream_found, stream_msg, current_stream = find_stream(client, stream_name)
    if stream_found:
        if current_stream.get('EncryptionType') == 'KMS':
            success, err_msg = stream_encryption_action(client, stream_name, action='stop_encryption', key_id=key_id, encryption_type=encryption_type, check_mode=check_mode)
            changed = success
            if wait:
                success, err_msg, results = wait_for_status(client, stream_name, 'ACTIVE', wait_timeout, check_mode=check_mode)
                if not success:
                    return (success, True, err_msg, results)
                err_msg = f'Kinesis Stream {stream_name} encryption stopped successfully.'
            else:
                err_msg = f'Stream {stream_name} is in the process of stopping encryption.'
        elif current_stream.get('EncryptionType') == 'NONE':
            success = True
            err_msg = f'Kinesis Stream {stream_name} encryption already stopped.'
    else:
        success = True
        changed = False
        err_msg = f'Stream {stream_name} does not exist.'
    if success:
        stream_found, stream_msg, results = find_stream(client, stream_name)
        tag_success, tag_msg, current_tags = get_tags(client, stream_name)
        if not current_tags:
            current_tags = dict()
        results = camel_dict_to_snake_dict(results)
        results['tags'] = current_tags
    return (success, changed, err_msg, results)