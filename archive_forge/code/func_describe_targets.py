from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_targets(connection, module, tg_arn, target=None):
    """
    Describe targets in a target group

    :param module: ansible module object
    :param connection: boto3 connection
    :param tg_arn: target group arn
    :param target: dictionary containing target id and port
    :return:
    """
    try:
        targets = describe_targets_with_backoff(connection, tg_arn, target)['TargetHealthDescriptions']
        if not targets:
            return {}
        return targets[0]
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg=f'Unable to describe target health for target {target}')