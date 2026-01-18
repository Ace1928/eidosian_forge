from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
def create_naptrrecord(self):
    record = 'naptrrecord %s -set ttl=%s;container=%s;order=%s;preference=%s;flags="%s";service="%s";replacement="%s"' % (self.dnsname, self.ttl, self.container, self.order, self.preference, self.flags, self.service, self.replacement)
    return record