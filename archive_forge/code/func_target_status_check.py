from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def target_status_check(connection, module, target_group_arn, target, target_status, target_status_timeout):
    reached_state = False
    timeout = target_status_timeout + time()
    while time() < timeout:
        health_state = describe_targets(connection, module, target_group_arn, target)['TargetHealth']['State']
        if health_state == target_status:
            reached_state = True
            break
        sleep(1)
    if not reached_state:
        module.fail_json(msg=f'Status check timeout of {target_status_timeout} exceeded, last status was {health_state}: ')