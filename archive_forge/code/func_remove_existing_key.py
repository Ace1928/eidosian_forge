from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from re import findall
def remove_existing_key(self, key_id):
    resp, info = fetch_url(self.module, '{0}/{1}'.format(self.url, key_id), headers=self.headers, method='DELETE')
    status_code = info['status']
    if status_code == 204:
        if self.state == 'absent':
            self.module.exit_json(changed=True, msg='Deploy key successfully deleted', id=key_id)
    else:
        self.handle_error(method='DELETE', info=info, key_id=key_id)