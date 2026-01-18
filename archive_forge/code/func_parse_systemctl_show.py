from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.facts.system.chroot import is_chroot
from ansible.module_utils.service import sysv_exists, sysv_is_enabled, fail_if_missing
from ansible.module_utils.common.text.converters import to_native
def parse_systemctl_show(lines):
    parsed = {}
    multival = []
    k = None
    for line in lines:
        if k is None:
            if '=' in line:
                k, v = line.split('=', 1)
                if k.startswith('Exec') and v.lstrip().startswith('{'):
                    if not v.rstrip().endswith('}'):
                        multival.append(v)
                        continue
                parsed[k] = v.strip()
                k = None
        else:
            multival.append(line)
            if line.rstrip().endswith('}'):
                parsed[k] = '\n'.join(multival).strip()
                multival = []
                k = None
    return parsed