from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def get_chain_policy(iptables_path, module, params):
    cmd = push_arguments(iptables_path, '-L', params, make_rule=False)
    if module.params['numeric']:
        cmd.append('--numeric')
    rc, out, err = module.run_command(cmd, check_rc=True)
    chain_header = out.split('\n')[0]
    result = re.search('\\(policy ([A-Z]+)\\)', chain_header)
    if result:
        return result.group(1)
    return None