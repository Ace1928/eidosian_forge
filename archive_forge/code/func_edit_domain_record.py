from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
import time
def edit_domain_record(self, record):
    if self.module.params.get('ip'):
        params = {'name': '@', 'data': self.module.params.get('ip')}
    if self.module.params.get('ip6'):
        params = {'name': '@', 'data': self.module.params.get('ip6')}
    resp = self.put('domains/%s/records/%s' % (self.domain_name, record['id']), data=params)
    status, json = self.jsonify(resp)
    return json['domain_record']