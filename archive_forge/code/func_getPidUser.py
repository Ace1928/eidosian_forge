from __future__ import (absolute_import, division, print_function)
import re
import platform
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def getPidUser(pid):
    ps_cmd = module.get_bin_path('ps', True)
    rc, ps_output, stderr = module.run_command([ps_cmd, '-o', 'user', '-p', str(pid)])
    user = ''
    if rc == 0:
        for line in ps_output.splitlines():
            if line != 'USER':
                user = line
    return user