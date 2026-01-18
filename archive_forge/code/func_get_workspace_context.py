from __future__ import absolute_import, division, print_function
import os
import json
import tempfile
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.six import integer_types
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_workspace_context(bin_path, project_path):
    workspace_ctx = {'current': 'default', 'all': []}
    command = [bin_path, 'workspace', 'list', '-no-color']
    rc, out, err = module.run_command(command, cwd=project_path)
    if rc != 0:
        module.warn('Failed to list Terraform workspaces:\n{0}'.format(err))
    for item in out.split('\n'):
        stripped_item = item.strip()
        if not stripped_item:
            continue
        elif stripped_item.startswith('* '):
            workspace_ctx['current'] = stripped_item.replace('* ', '')
            workspace_ctx['all'].append(stripped_item.replace('* ', ''))
        else:
            workspace_ctx['all'].append(stripped_item)
    return workspace_ctx