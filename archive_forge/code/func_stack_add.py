from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def stack_add(self, result):
    data = dict()
    changed = False
    data['cmd'] = 'add host {0} rack={1} rank={2} appliance={3}'.format(self.hostname, self.rack, self.rank, self.appliance)
    self.do_request(self.endpoint, payload=json.dumps(data), headers=self.header, method='POST')
    self.stack_sync()
    result['changed'] = changed
    result['stdout'] = 'api call successful'.rstrip('\r\n')