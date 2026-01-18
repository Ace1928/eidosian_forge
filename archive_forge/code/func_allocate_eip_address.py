import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def allocate_eip_address(client, module):
    """Release an EIP from your EIP Pool
    Args:
        client (botocore.client.EC2): Boto3 client
        module: AnsibleAWSModule class instance

    Basic Usage:
        >>> client = boto3.client('ec2')
        >>> module = AnsibleAWSModule(...)
        >>> allocate_eip_address(client, module)
        (
            True, '', ''
        )

    Returns:
        Tuple (bool, str, str)
    """
    new_eip = None
    msg = ''
    params = {'Domain': 'vpc'}
    if module.check_mode:
        ip_allocated = True
        new_eip = None
        return (ip_allocated, msg, new_eip)
    try:
        new_eip = client.allocate_address(aws_retry=True, **params)['AllocationId']
        ip_allocated = True
        msg = f'eipalloc id {new_eip} created'
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e)
    return (ip_allocated, msg, new_eip)