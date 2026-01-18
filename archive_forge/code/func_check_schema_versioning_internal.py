from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def check_schema_versioning_internal(results, trace, schema, params, version):
    if not schema or not params:
        return
    matched = __check_if_system_version_is_supported(schema['v_range'] if 'v_range' in schema else {}, version)
    if matched['supported'] is False:
        results['mismatches'].append('option %s %s' % (__concat_attribute_sequence(trace), matched['reason']))
    if 'type' not in schema:
        return
    if schema['type'] == 'list':
        if type(params) is not list:
            raise AssertionError()
        if 'children' in schema:
            if 'options' in schema:
                raise AssertionError()
            for list_item in params:
                if type(list_item) is not dict:
                    results['mismatches'].append("option [%s]' playload is inconsistent with schema." % __concat_attribute_sequence(trace))
                    continue
                for key in list_item:
                    value = list_item[key]
                    key_string = '%s(%s)' % (key, value) if type(value) in [int, bool, str] else key
                    trace.append(key_string)
                    check_schema_versioning_internal(results, trace, schema['children'][key], value, version)
                    del trace[-1]
        elif 'options' in schema:
            for param in params:
                if type(param) not in [int, bool, str]:
                    raise AssertionError()
                target_option = None
                for option in schema['options']:
                    if option['value'] == param:
                        target_option = option
                        break
                if not target_option:
                    raise AssertionError()
                trace.append('[%s]' % param)
                check_schema_versioning_internal(results, trace, target_option, param, version)
                del trace[-1]
    elif schema['type'] == 'dict':
        if type(params) is not dict:
            raise AssertionError()
        if 'children' in schema:
            for dict_item_key in params:
                dict_item_value = params[dict_item_key]
                if dict_item_key not in schema['children']:
                    raise AssertionError()
                key_string = '%s(%s)' % (dict_item_key, dict_item_value) if type(dict_item_value) in [int, bool, str] else dict_item_key
                trace.append(key_string)
                check_schema_versioning_internal(results, trace, schema['children'][dict_item_key], dict_item_value, version)
                del trace[-1]
    elif type(params) not in [int, str, bool]:
        raise AssertionError()