from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_params(module, params, nodegroup):
    for param in ['nodeRole', 'subnets', 'diskSize', 'instanceTypes', 'amiTypes', 'remoteAccess', 'capacityType']:
        if param in nodegroup and param in params:
            if nodegroup[param] != params[param]:
                module.fail_json(msg=f'Cannot modify parameter {param}.')
    if 'launchTemplate' not in nodegroup and 'launchTemplate' in params:
        module.fail_json(msg='Cannot add Launch Template in this Nodegroup.')
    if nodegroup['updateConfig'] != params['updateConfig']:
        return True
    if nodegroup['scalingConfig'] != params['scalingConfig']:
        return True
    return False