from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def do_member_operation(self, path, name):
    toplevel_name = (path + '_' + name).replace('-', '_').replace('.', '_').replace('+', 'plus')
    data = self._module.params
    if not data['member_state']:
        return
    if not data['member_path']:
        self._module.fail_json('member_path is empty while member_state is %s' % data['member_state'])
    attribute_path = list()
    for attr in data['member_path'].split('/'):
        if attr == '':
            continue
        attribute_path.append(attr.strip(' '))
    if not len(attribute_path):
        raise AssertionError('member_path should have at least one attribute')
    state_present = 'state' in data
    if state_present and (not self._mkeyname):
        raise AssertionError('Invalid mkey scheme!')
    if state_present and (not data[toplevel_name] or not data[toplevel_name][self._mkeyname]):
        raise AssertionError('parameter %s or %s.%s empty!' % (toplevel_name, toplevel_name, self._mkeyname))
    toplevel_url_token = ''
    if state_present:
        toplevel_url_token = '/%s' % data[toplevel_name][self._mkeyname]
    arg_spec = self._module.argument_spec[toplevel_name]['options']
    attr_spec = arg_spec
    attr_params = data[toplevel_name]
    if not attr_params:
        raise AssertionError('Parameter %s is empty' % toplevel_name)
    attr_blobs = list()
    for attr_pair in attribute_path:
        attr_pair_split = attr_pair.split(':')
        attr = attr_pair_split[0]
        if attr not in attr_spec:
            self._module.fail_json('Attribute %s not as part of module schema' % attr)
        attr_spec = attr_spec[attr]
        attr_type = attr_spec['type']
        if len(attr_pair_split) != 2 and attr_type != 'dict':
            self._module.fail_json('wrong attribute format: %s' % attr_pair)
        attr_mkey = attr_pair_split[1] if attr_type == 'list' else None
        if 'options' not in attr_spec:
            raise AssertionError('Attribute %s not member operable, no children options' % attr)
        attr_blob = dict()
        attr_blob['name'] = attr
        attr_blob['mkey'] = attr_mkey
        attr_blob['schema'] = attr_spec['options']
        attr_spec = attr_spec['options']
        attr_blobs.append(attr_blob)
    trace = list()
    trace_param = list()
    trace_url_tokens = list()
    urls = list()
    results = list()
    trace.append(toplevel_name)
    self._validate_member_parameter(trace, trace_param, trace_url_tokens, attr_blobs, attr_params[attr_blobs[0]['name']])
    self._process_sub_object(urls, toplevel_url_token, trace_url_tokens, path, name)
    for sub_obj in urls:
        result = self._request_sub_object(sub_obj)
        results.append((sub_obj, result))
    self._process_sub_object_result(results)