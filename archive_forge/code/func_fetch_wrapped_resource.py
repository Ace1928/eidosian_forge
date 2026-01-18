from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def fetch_wrapped_resource(module, kind, wrap_kind, wrap_path):
    result = fetch_resource(module, self_link(module), wrap_kind)
    if result is None or wrap_path not in result:
        return None
    result = unwrap_resource(result[wrap_path], module)
    if result is None:
        return None
    if result['kind'] != kind:
        module.fail_json(msg='Incorrect result: {kind}'.format(**result))
    return result