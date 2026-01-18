from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def check_schema_versioning(fos, versioned_schema, top_level_param):
    trace = list()
    results = dict()
    results['matched'] = True
    results['mismatches'] = list()
    system_version = fos._conn.get_system_version()
    params = fos._module.params[top_level_param]
    results['system_version'] = system_version
    if not params:
        return results
    v_range = versioned_schema['v_range']
    module_matched = __check_if_system_version_is_supported(v_range, system_version)
    if not module_matched['supported']:
        results['matched'] = False
        results['mismatches'].append('module fortios_%s %s' % (top_level_param, module_matched['reason']))
        return results
    for param_name in params:
        param_value = params[param_name]
        if not param_value or param_name not in versioned_schema['children']:
            continue
        key_string = '%s(%s)' % (param_name, param_value) if type(param_value) in [int, bool, str] else param_name
        trace.append(key_string)
        check_schema_versioning_internal(results, trace, versioned_schema['children'][param_name], param_value, system_version)
        del trace[-1]
    if len(results['mismatches']):
        results['matched'] = False
    return results