from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text, to_native
def export_manifest(module, manifest):
    path = '/subscription/consumers/%s/export' % manifest['uuid']
    try:
        resp, info = fetch_portal(module, path, 'GET', accept_header='application/zip')
        if not module.check_mode:
            with open(module.params['path'], 'wb') as f:
                while True:
                    data = resp.read(65536)
                    if not data:
                        break
                    f.write(data)
    except Exception as e:
        module.fail_json(msg='Failure downloading manifest, {0}'.format(to_native(e)))