from .botocore import is_boto3_error_code
from .retries import AWSRetry
def get_elb(connection, module, elb_name):
    """
    Get an ELB based on name. If not found, return None.

    :param connection: AWS boto3 elbv2 connection
    :param module: Ansible module
    :param elb_name: Name of load balancer to get
    :return: boto3 ELB dict or None if not found
    """
    try:
        return _get_elb(connection, module, elb_name)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e)