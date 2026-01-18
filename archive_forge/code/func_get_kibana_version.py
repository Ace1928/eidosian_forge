from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_kibana_version(module, plugin_bin, allow_root):
    cmd_args = [plugin_bin, '--version']
    if allow_root:
        cmd_args.append('--allow-root')
    rc, out, err = module.run_command(cmd_args)
    if rc != 0:
        module.fail_json(msg='Failed to get Kibana version : %s' % err)
    return out.strip()