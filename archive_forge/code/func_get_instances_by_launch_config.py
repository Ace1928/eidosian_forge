import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def get_instances_by_launch_config(props, lc_check, initial_instances):
    new_instances = []
    old_instances = []
    if lc_check:
        for i in props['instances']:
            if 'launch_template' in props['instance_facts'][i]:
                old_instances.append(i)
            elif props['instance_facts'][i].get('launch_config_name') == props['launch_config_name']:
                new_instances.append(i)
            else:
                old_instances.append(i)
    else:
        module.debug(f'Comparing initial instances with current: {(*initial_instances,)}')
        for i in props['instances']:
            if i not in initial_instances:
                new_instances.append(i)
            else:
                old_instances.append(i)
    module.debug(f'New instances: {len(new_instances)}, {(*new_instances,)}')
    module.debug(f'Old instances: {len(old_instances)}, {(*old_instances,)}')
    return (new_instances, old_instances)