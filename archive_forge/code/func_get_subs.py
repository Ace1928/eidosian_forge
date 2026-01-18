from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text, to_native
def get_subs(module, manifest):
    path = '/subscription/consumers/%s/entitlements' % manifest['uuid']
    resp, info = fetch_portal(module, path, 'GET')
    all_subs = json.loads(to_text(resp.read()))
    subs = [s for s in all_subs if s['pool']['id'] == module.params['pool_id']]
    return subs