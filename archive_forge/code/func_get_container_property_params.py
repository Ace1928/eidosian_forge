from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import cc
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_container_property_params():
    return ('image', 'vcpus', 'memory', 'command', 'job_role_arn', 'volumes', 'environment', 'mount_points', 'readonly_root_filesystem', 'privileged', 'ulimits', 'user')