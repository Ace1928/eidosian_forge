from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_domain_byid(self, dom_id):
    return self.conn.storagedomains.get(id=dom_id)