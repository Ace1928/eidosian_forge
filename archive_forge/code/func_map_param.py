from __future__ import absolute_import, division, print_function
import copy
import json
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def map_param(self, k, v, is_update):

    def helper(item):
        return {camel_case_key(k): v for k, v in item.items()}

    def needs_camel_case(k):
        spec = self._module.argument_spec[k]
        return spec.get('type') == 'list' and spec.get('elements') == 'dict' and spec.get('options') or (spec.get('type') == 'dict' and spec.get('options'))
    if k in self.api_params and v is not None:
        if isinstance(v, dict) and needs_camel_case(k):
            v = helper(v)
        elif isinstance(v, (list, tuple)) and needs_camel_case(k):
            v = [helper(i) for i in v]
        if is_update and k in self.create_only_fields:
            return
        return (camel_case_key(k), v)