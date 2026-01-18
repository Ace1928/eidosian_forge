from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def push_arguments(iptables_path, action, params, make_rule=True):
    cmd = [iptables_path]
    cmd.extend(['-t', params['table']])
    cmd.extend([action, params['chain']])
    if action == '-I' and params['rule_num']:
        cmd.extend([params['rule_num']])
    if make_rule:
        cmd.extend(construct_rule(params))
    return cmd