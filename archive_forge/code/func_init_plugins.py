from __future__ import absolute_import, division, print_function
import os
import json
import tempfile
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.six import integer_types
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def init_plugins(bin_path, project_path, backend_config, backend_config_files, init_reconfigure, provider_upgrade, plugin_paths, workspace):
    command = [bin_path, 'init', '-input=false', '-no-color']
    if backend_config:
        for key, val in backend_config.items():
            command.extend(['-backend-config', '{0}={1}'.format(key, val)])
    if backend_config_files:
        for f in backend_config_files:
            command.extend(['-backend-config', f])
    if init_reconfigure:
        command.extend(['-reconfigure'])
    if provider_upgrade:
        command.extend(['-upgrade'])
    if plugin_paths:
        for plugin_path in plugin_paths:
            command.extend(['-plugin-dir', plugin_path])
    rc, out, err = module.run_command(command, check_rc=True, cwd=project_path, environ_update={'TF_WORKSPACE': workspace})