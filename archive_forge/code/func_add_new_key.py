from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from re import findall
def add_new_key(self):
    request_body = {'title': self.name, 'key': self.key, 'read_only': self.read_only}
    resp, info = fetch_url(self.module, self.url, data=self.module.jsonify(request_body), headers=self.headers, method='POST', timeout=30)
    status_code = info['status']
    if status_code == 201:
        response_body = self.module.from_json(resp.read())
        key_id = response_body['id']
        self.module.exit_json(changed=True, msg='Deploy key successfully added', id=key_id)
    elif status_code == 422:
        self.module.exit_json(changed=False, msg='Deploy key already exists')
    else:
        self.handle_error(method='POST', info=info)