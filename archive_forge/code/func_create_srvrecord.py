from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
def create_srvrecord(self):
    record = 'srvrecord %s -set ttl=%s;container=%s;priority=%s;weight=%s;port=%s;target=%s' % (self.dnsname, self.ttl, self.container, self.priority, self.weight, self.port, self.target)
    return record