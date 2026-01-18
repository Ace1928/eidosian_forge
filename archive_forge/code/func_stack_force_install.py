from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def stack_force_install(self, result):
    data = {'cmd': 'set host boot {0} action=install'.format(self.hostname)}
    self.do_request(self.endpoint, payload=json.dumps(data), headers=self.header, method='POST')
    changed = True
    self.stack_sync()
    result['changed'] = changed
    result['stdout'] = 'api call successful'.rstrip('\r\n')