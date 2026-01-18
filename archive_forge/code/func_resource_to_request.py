from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest
import json
import re
def resource_to_request(module):
    request = {u'name': name_pattern(module.params.get('name'), module)}
    return_vals = {}
    for k, v in request.items():
        if v or v is False:
            return_vals[k] = v
    return return_vals