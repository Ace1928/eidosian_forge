from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def check_parted_label(device):
    """
    Determines if parted needs a label to complete its duties. Versions prior
    to 3.1 don't return data when there is no label. For more information see:
    http://upstream.rosalinux.ru/changelogs/libparted/3.1/changelog.html
    """
    global parted_exec
    parted_major, parted_minor, dummy = parted_version()
    if parted_major == 3 and parted_minor >= 1 or parted_major > 3:
        return False
    rc, out, err = module.run_command('%s -s -m %s print' % (parted_exec, device))
    if rc != 0 and 'unrecognised disk label' in out.lower():
        return True
    return False