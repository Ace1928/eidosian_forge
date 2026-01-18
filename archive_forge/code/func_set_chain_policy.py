from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def set_chain_policy(iptables_path, module, params):
    cmd = push_arguments(iptables_path, '-P', params, make_rule=False)
    cmd.append(params['policy'])
    module.run_command(cmd, check_rc=True)