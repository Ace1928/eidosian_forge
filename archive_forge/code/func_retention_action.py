import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def retention_action(client, stream_name, retention_period=24, action='increase', check_mode=False):
    """Increase or Decrease the retention of messages in the Kinesis stream.
    Args:
        client (botocore.client.EC2): Boto3 client.
        stream_name (str): The name of the kinesis stream.

    Kwargs:
        retention_period (int): This is how long messages will be kept before
            they are discarded. This can not be less than 24 hours.
        action (str): The action to perform.
            valid actions == create and delete
            default=create
        check_mode (bool): This will pass DryRun as one of the parameters to the aws api.
            default=False

    Basic Usage:
        >>> client = boto3.client('kinesis')
        >>> stream_name = 'test-stream'
        >>> retention_period = 48
        >>> retention_action(client, stream_name, retention_period, action='increase')

    Returns:
        Tuple (bool, str)
    """
    success = False
    err_msg = ''
    params = {'StreamName': stream_name}
    try:
        if not check_mode:
            if action == 'increase':
                params['RetentionPeriodHours'] = retention_period
                client.increase_stream_retention_period(**params)
                success = True
                err_msg = f'Retention Period increased successfully to {retention_period}'
            elif action == 'decrease':
                params['RetentionPeriodHours'] = retention_period
                client.decrease_stream_retention_period(**params)
                success = True
                err_msg = f'Retention Period decreased successfully to {retention_period}'
            else:
                err_msg = f'Invalid action {action}'
        elif action == 'increase':
            success = True
        elif action == 'decrease':
            success = True
        else:
            err_msg = f'Invalid action {action}'
    except botocore.exceptions.ClientError as e:
        err_msg = to_native(e)
    return (success, err_msg)