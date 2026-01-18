from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def image_exists(module, executable, name):
    command = [executable, 'image', 'exists', name]
    rc, out, err = module.run_command(command)
    if rc == 1:
        return False
    elif 'Command "exists" not found' in err:
        command = [executable, 'image', 'ls', '-q', name]
        rc2, out2, err2 = module.run_command(command)
        if rc2 != 0:
            return False
    return True