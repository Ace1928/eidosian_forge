from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def build_result(self, result, domain):
    if result is None:
        return None
    res = {}
    for k in self.attribute_map:
        v = result.get(self.attribute_map[k], None)
        if v is not None:
            if k == 'record' and v == '@':
                v = ''
            res[k] = v
    res['domain'] = domain
    return res