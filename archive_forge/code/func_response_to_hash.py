from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest
import json
import re
def response_to_hash(module, response):
    return {u'name': name_pattern(module.params.get('name'), module), u'url': response.get(u'url'), u'size': response.get(u'size')}