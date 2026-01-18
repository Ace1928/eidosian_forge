from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
def create_arecord(self):
    if self.dnstype == 'AAAA':
        record = 'aaaarecord %s %s -set ttl=%s;container=%s' % (self.dnsname, self.address, self.ttl, self.container)
    else:
        record = 'arecord %s %s -set ttl=%s;container=%s' % (self.dnsname, self.address, self.ttl, self.container)
    return record