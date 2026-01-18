from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def get_iptables_version(iptables_path, module):
    cmd = [iptables_path, '--version']
    rc, out, err = module.run_command(cmd, check_rc=True)
    return out.split('v')[1].rstrip('\n')