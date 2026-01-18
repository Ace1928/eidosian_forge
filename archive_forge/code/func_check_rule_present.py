from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def check_rule_present(iptables_path, module, params):
    cmd = push_arguments(iptables_path, '-C', params)
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    return rc == 0