from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def is_activated(self):
    rc, out, dummy = self._beadm_list()
    if rc == 0:
        line = self._find_be_by_name(out)
        if line is None:
            return False
        if self.is_freebsd:
            if 'R' in line[1]:
                return True
        elif 'R' in line[2]:
            return True
    return False