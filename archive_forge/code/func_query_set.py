from __future__ import absolute_import, division, print_function
import os
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native
def query_set(module, set, action):
    system_sets = ['@live-rebuild', '@module-rebuild', '@preserved-rebuild', '@security', '@selected', '@system', '@world', '@x11-module-rebuild']
    if set in system_sets:
        if action == 'unmerge':
            module.fail_json(msg='set %s cannot be removed' % set)
        return False
    world_sets_path = '/var/lib/portage/world_sets'
    if not os.path.exists(world_sets_path):
        return False
    cmd = 'grep %s %s' % (set, world_sets_path)
    rc, out, err = module.run_command(cmd)
    return rc == 0