from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
import time
def create_domain_record(self):
    if self.module.params.get('ip'):
        params = {'name': '@', 'type': 'A', 'data': self.module.params.get('ip')}
    if self.module.params.get('ip6'):
        params = {'name': '@', 'type': 'AAAA', 'data': self.module.params.get('ip6')}
    resp = self.post('domains/%s/records' % self.domain_name, data=params)
    status, json = self.jsonify(resp)
    return json['domain_record']