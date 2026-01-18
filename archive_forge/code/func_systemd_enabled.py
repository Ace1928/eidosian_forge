from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def systemd_enabled(self):
    try:
        f = open('/proc/1/comm', 'r')
    except IOError:
        return False
    for line in f:
        if 'systemd' in line:
            return True
    return False