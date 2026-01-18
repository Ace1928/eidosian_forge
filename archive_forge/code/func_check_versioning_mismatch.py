from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def check_versioning_mismatch(self, track, schema, params):
    if not params or not schema:
        return
    param_type = schema['type'] if 'type' in schema else None
    v_range = schema['v_range'] if 'v_range' in schema else None
    matched, checking_message = self._version_matched(v_range)
    if not matched:
        param_path = track[0]
        for _param in track[1:]:
            param_path += '-->%s' % _param
        self.version_check_warnings.append('param: %s %s' % (param_path, checking_message))
    if param_type == 'dict' and 'options' in schema:
        if not isinstance(params, dict):
            raise AssertionError()
        for sub_param_key in params:
            sub_param = params[sub_param_key]
            if sub_param_key in schema['options']:
                sub_schema = schema['options'][sub_param_key]
                track.append(sub_param_key)
                self.check_versioning_mismatch(track, sub_schema, sub_param)
                del track[-1]
    elif param_type == 'list' and 'options' in schema:
        if not isinstance(params, list):
            raise AssertionError()
        for grouped_param in params:
            if not isinstance(grouped_param, dict):
                raise AssertionError()
            for sub_param_key in grouped_param:
                sub_param = grouped_param[sub_param_key]
                if sub_param_key in schema['options']:
                    sub_schema = schema['options'][sub_param_key]
                    track.append(sub_param_key)
                    self.check_versioning_mismatch(track, sub_schema, sub_param)
                    del track[-1]