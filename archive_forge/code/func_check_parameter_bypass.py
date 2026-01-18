from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def check_parameter_bypass(schema, module_level2_name):
    schema = modify_argument_spec(schema)
    params = _load_params()
    if not params:
        return schema
    if 'bypass_validation' in params and params['bypass_validation'] is True:
        top_level_schema = dict()
        for key in schema:
            if key != module_level2_name:
                top_level_schema[key] = schema[key]
            elif not params[module_level2_name] or type(params[module_level2_name]) is dict:
                top_level_schema[module_level2_name] = dict()
                top_level_schema[module_level2_name]['required'] = False
                top_level_schema[module_level2_name]['type'] = 'dict'
            elif type(params[module_level2_name]) is list:
                top_level_schema[module_level2_name] = dict()
                top_level_schema[module_level2_name]['required'] = False
                top_level_schema[module_level2_name]['type'] = 'list'
            else:
                raise Exception('Value of %s must be a dict or list' % module_level2_name)
        return top_level_schema
    return schema