from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest
import json
import re
def fetch_resource(module, link, allow_not_found=True):
    auth = GcpSession(module, 'sourcerepo')
    return return_if_object(module, auth.get(link), allow_not_found)