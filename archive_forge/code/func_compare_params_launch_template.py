from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_params_launch_template(module, params, nodegroup):
    if 'launchTemplate' not in params:
        module.fail_json(msg='Cannot exclude Launch Template in this Nodegroup.')
    else:
        for key in ['name', 'id']:
            if key in params['launchTemplate'] and params['launchTemplate'][key] != nodegroup['launchTemplate'][key]:
                module.fail_json(msg=f'Cannot modify Launch Template {key}.')
        if 'version' in params['launchTemplate'] and params['launchTemplate']['version'] != nodegroup['launchTemplate']['version']:
            return True
    return False