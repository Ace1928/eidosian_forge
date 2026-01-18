from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text, to_native
def remove_sub(module, manifest, sub):
    path = '/subscription/consumers/%s/entitlements/%s' % (manifest['uuid'], sub['id'])
    fetch_portal(module, path, 'DELETE')