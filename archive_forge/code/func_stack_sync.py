from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def stack_sync(self):
    self.do_request(self.endpoint, payload=json.dumps({'cmd': 'sync config'}), headers=self.header, method='POST')
    self.do_request(self.endpoint, payload=json.dumps({'cmd': 'sync host config'}), headers=self.header, method='POST')