from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def build_launch_specification(launch_spec):
    """
    Remove keys that have a value of None from Launch Specification
    Descend into these subkeys:
    network_interfaces
    block_device_mappings
    monitoring
    placement
    iam_instance_profile
    """
    assigned_keys = dict(((k, v) for k, v in launch_spec.items() if v is not None))
    sub_key_to_build = ['placement', 'iam_instance_profile', 'monitoring']
    for subkey in sub_key_to_build:
        if launch_spec[subkey] is not None:
            assigned_keys[subkey] = dict(((k, v) for k, v in launch_spec[subkey].items() if v is not None))
    if launch_spec['network_interfaces'] is not None:
        interfaces = []
        for iface in launch_spec['network_interfaces']:
            interfaces.append(dict(((k, v) for k, v in iface.items() if v is not None)))
        assigned_keys['network_interfaces'] = interfaces
    if launch_spec['block_device_mappings'] is not None:
        block_devs = []
        for dev in launch_spec['block_device_mappings']:
            block_devs.append(dict(((k, v) for k, v in dev.items() if v is not None)))
        assigned_keys['block_device_mappings'] = block_devs
    return snake_dict_to_camel_dict(assigned_keys, capitalize_first=True)