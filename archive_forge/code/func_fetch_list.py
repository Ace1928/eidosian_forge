from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule
import json
def fetch_list(module, link):
    auth = GcpSession(module, 'cloudbuild')
    return auth.list(link, return_if_object, array_name='triggers')