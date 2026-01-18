from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def stack_remove(self, result):
    data = dict()
    data['cmd'] = 'remove host {0}'.format(self.hostname)
    self.do_request(self.endpoint, payload=json.dumps(data), headers=self.header, method='POST')
    self.stack_sync()
    result['changed'] = True
    result['stdout'] = 'api call successful'.rstrip('\r\n')