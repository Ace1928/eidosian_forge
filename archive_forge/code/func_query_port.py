from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def query_port(module, port_path, name, state='present'):
    """ Returns whether a port is installed or not. """
    if state == 'present':
        rc, out, err = module.run_command([port_path, '-q', 'installed', name])
        if rc == 0 and out.strip().startswith(name + ' '):
            return True
        return False
    elif state == 'active':
        rc, out, err = module.run_command([port_path, '-q', 'installed', name])
        if rc == 0 and '(active)' in out:
            return True
        return False