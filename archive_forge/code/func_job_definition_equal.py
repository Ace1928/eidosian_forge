from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.batch import cc
from ansible_collections.amazon.aws.plugins.module_utils.batch import set_api_params
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def job_definition_equal(module, current_definition):
    equal = True
    for param in get_base_params():
        if module.params.get(param) != current_definition.get(cc(param)):
            equal = False
            break
    for param in get_container_property_params():
        if module.params.get(param) != current_definition.get('containerProperties').get(cc(param)):
            equal = False
            break
    for param in get_retry_strategy_params():
        if module.params.get(param) != current_definition.get('retryStrategy').get(cc(param)):
            equal = False
            break
    return equal