from .botocore import is_boto3_error_code
from .retries import AWSRetry
def get_elb_listener_rules(connection, module, listener_arn):
    """
    Get rules for a particular ELB listener using the listener ARN.

    :param connection: AWS boto3 elbv2 connection
    :param module: Ansible module
    :param listener_arn: ARN of the ELB listener
    :return: boto3 ELB rules list
    """
    try:
        return AWSRetry.jittered_backoff()(connection.describe_rules)(ListenerArn=listener_arn)['Rules']
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e)