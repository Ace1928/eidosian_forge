from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def schema_to_module_spec(schema):
    rdata = dict()
    if 'type' not in schema:
        raise AssertionError('Invalid Schema')
    if schema['type'] == 'dict' or (schema['type'] == 'list' and 'children' in schema):
        if 'children' not in schema:
            raise AssertionError()
        rdata['type'] = schema['type']
        if schema['type'] == 'list':
            rdata['elements'] = schema.get('elements')
        rdata['required'] = schema['required'] if 'required' in schema else False
        rdata['options'] = dict()
        for child in schema['children']:
            child_value = schema['children'][child]
            rdata['options'][child] = schema_to_module_spec(child_value)
            if is_secret_field(child):
                rdata['options'][child]['no_log'] = True
    elif schema['type'] in ['integer', 'string'] or (schema['type'] == 'list' and 'children' not in schema):
        if schema['type'] == 'integer':
            rdata['type'] = 'int'
        elif schema['type'] == 'string':
            rdata['type'] = 'str'
        elif schema['type'] == 'list':
            rdata['type'] = 'list'
            rdata['elements'] = schema.get('elements')
        else:
            raise AssertionError()
        rdata['required'] = schema['required'] if 'required' in schema else False
        if 'options' in schema:
            rdata['choices'] = [option['value'] for option in schema['options']]
    else:
        raise AssertionError()
    return rdata