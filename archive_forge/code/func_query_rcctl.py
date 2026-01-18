from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def query_rcctl(self, cmd):
    svcs = []
    rc, stdout, stderr = self.module.run_command('%s ls %s' % (self.rcctl_path, cmd))
    if 'needs root privileges' in stderr.lower():
        self.module.warn('rcctl requires root privileges')
    else:
        for svc in stdout.split('\n'):
            if svc == '':
                continue
            else:
                svcs.append(svc)
    return svcs