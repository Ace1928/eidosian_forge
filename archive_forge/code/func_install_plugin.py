from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def install_plugin(module, plugin_bin, plugin_name, url, timeout, allow_root, kibana_version='4.6'):
    if LooseVersion(kibana_version) > LooseVersion('4.6'):
        kibana_plugin_bin = os.path.join(os.path.dirname(plugin_bin), 'kibana-plugin')
        cmd_args = [kibana_plugin_bin, 'install']
        if url:
            cmd_args.append(url)
        else:
            cmd_args.append(plugin_name)
    else:
        cmd_args = [plugin_bin, 'plugin', PACKAGE_STATE_MAP['present'], plugin_name]
        if url:
            cmd_args.extend(['--url', url])
    if timeout:
        cmd_args.extend(['--timeout', timeout])
    if allow_root:
        cmd_args.append('--allow-root')
    if module.check_mode:
        return (True, ' '.join(cmd_args), 'check mode', '')
    rc, out, err = module.run_command(cmd_args)
    if rc != 0:
        reason = parse_error(out)
        module.fail_json(msg=reason)
    return (True, ' '.join(cmd_args), out, err)