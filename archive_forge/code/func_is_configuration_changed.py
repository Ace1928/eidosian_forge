from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def is_configuration_changed(module, current):
    """
    compare configuration's description and properties
    python 2.7+ version:
    prop_module = {str(k): str(v) for k, v in module.params.get("config").items()}
    """
    prop_module = {}
    for k, v in module.params.get('config').items():
        prop_module[str(k)] = str(v)
    if prop_to_dict(current.get('ServerProperties', '')) == prop_module:
        if current.get('Description', '') == module.params.get('description'):
            return False
    return True