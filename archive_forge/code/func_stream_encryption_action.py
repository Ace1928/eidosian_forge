import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def stream_encryption_action(client, stream_name, action='start_encryption', encryption_type='', key_id='', timeout=300, check_mode=False):
    """Create, Encrypt or Delete an Amazon Kinesis Stream.
    Args:
        client (botocore.client.EC2): Boto3 client.
        stream_name (str): The name of the kinesis stream.

    Kwargs:
        shard_count (int): Number of shards this stream will use.
        action (str): The action to perform.
            valid actions == create and delete
            default=create
        encryption_type (str): NONE or KMS
        key_id (str): The GUID or alias for the KMS key
        check_mode (bool): This will pass DryRun as one of the parameters to the aws api.
            default=False

    Basic Usage:
        >>> client = boto3.client('kinesis')
        >>> stream_name = 'test-stream'
        >>> shard_count = 20
        >>> stream_action(client, stream_name, shard_count, action='create', encryption_type='KMS',key_id='alias/aws')

    Returns:
        List (bool, str)
    """
    success = False
    err_msg = ''
    params = {'StreamName': stream_name}
    try:
        if not check_mode:
            if action == 'start_encryption':
                params['EncryptionType'] = encryption_type
                params['KeyId'] = key_id
                client.start_stream_encryption(**params)
                success = True
            elif action == 'stop_encryption':
                params['EncryptionType'] = encryption_type
                params['KeyId'] = key_id
                client.stop_stream_encryption(**params)
                success = True
            else:
                err_msg = f'Invalid encryption action {action}'
        elif action == 'start_encryption':
            success = True
        elif action == 'stop_encryption':
            success = True
        else:
            err_msg = f'Invalid encryption action {action}'
    except botocore.exceptions.ClientError as e:
        err_msg = to_native(e)
    return (success, err_msg)