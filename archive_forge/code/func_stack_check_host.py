from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def stack_check_host(self):
    res = self.do_request(self.endpoint, payload=json.dumps({'cmd': 'list host'}), headers=self.header, method='POST')
    return self.hostname in res.read()