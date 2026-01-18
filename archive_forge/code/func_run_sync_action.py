from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
def run_sync_action(self):
    post_body = dict()
    if self.state == 'running':
        if self.current_state == 'idle':
            if self.delete_recovery_point:
                post_body.update(dict(deleteRecoveryPointIfNecessary=self.delete_recovery_point))
            suffix = 'sync'
        else:
            suffix = 'resume'
    else:
        suffix = 'suspend'
    endpoint = self.url + '/storage-systems/%s/async-mirrors/%s/%s' % (self.ssid, self.amg_id, suffix)
    rc, resp = request(endpoint, method='POST', url_username=self.user, url_password=self.pwd, validate_certs=self.certs, data=json.dumps(post_body), headers=self.post_headers, ignore_errors=True)
    if not str(rc).startswith('2'):
        self.module.fail_json(msg=str(resp['errorMessage']))
    return resp