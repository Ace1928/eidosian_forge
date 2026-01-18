from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
import time
def domain_record(self):
    resp = self.get('domains/%s' % self.domain_name)
    status, json = self.jsonify(resp)
    return json