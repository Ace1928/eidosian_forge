import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def enforce_required_arguments_for_create():
    """As many arguments are not required for autoscale group deletion
    they cannot be mandatory arguments for the module, so we enforce
    them here"""
    missing_args = []
    if module.params.get('launch_config_name') is None and module.params.get('launch_template') is None:
        module.fail_json(msg='Missing either launch_config_name or launch_template for autoscaling group create')
    for arg in ('min_size', 'max_size'):
        if module.params[arg] is None:
            missing_args.append(arg)
    if missing_args:
        module.fail_json(msg=f'Missing required arguments for autoscaling group create: {','.join(missing_args)}')