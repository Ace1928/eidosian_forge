import time
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
def list_purgeable_instances(props, lc_check, lt_check, replace_instances, initial_instances):
    instances_to_terminate = []
    instances = (inst_id for inst_id in replace_instances if inst_id in props['instances'])
    if 'launch_config_name' in module.params:
        if lc_check:
            for i in instances:
                if 'launch_template' in props['instance_facts'][i] or props['instance_facts'][i]['launch_config_name'] != props['launch_config_name']:
                    instances_to_terminate.append(i)
        else:
            for i in instances:
                if i in initial_instances:
                    instances_to_terminate.append(i)
    elif 'launch_template' in module.params:
        if lt_check:
            for i in instances:
                if 'launch_config_name' in props['instance_facts'][i] or props['instance_facts'][i]['launch_template'] != props['launch_template']:
                    instances_to_terminate.append(i)
        else:
            for i in instances:
                if i in initial_instances:
                    instances_to_terminate.append(i)
    return instances_to_terminate