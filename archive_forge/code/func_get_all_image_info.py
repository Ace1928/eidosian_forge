from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def get_all_image_info(module, executable):
    command = [executable, 'image', 'ls', '-q']
    rc, out, err = module.run_command(command)
    out = out.strip()
    if out:
        name = out.split('\n')
        res = get_image_info(module, executable, name)
        return res
    return json.dumps([])