from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def make_current_modifiable_param_dict(module, conn, name):
    """Gets the current state of the cache parameter group and creates a dict with the format: {ParameterName: [Allowed_Values, DataType, ParameterValue]}"""
    current_info = get_info(conn, name)
    if current_info is False:
        module.fail_json(msg=f'Could not connect to the cache parameter group {name}.')
    parameters = current_info['Parameters']
    modifiable_params = {}
    for param in parameters:
        if param['IsModifiable']:
            modifiable_params[param['ParameterName']] = [param.get('AllowedValues')]
            modifiable_params[param['ParameterName']].append(param['DataType'])
            modifiable_params[param['ParameterName']].append(param.get('ParameterValue'))
    return modifiable_params