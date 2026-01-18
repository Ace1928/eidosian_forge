from __future__ import absolute_import, division, print_function
import os
import platform
import re
import shlex
import sqlite3
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def package_absent(names, pkg_spec, module):
    remove_cmd = 'pkg_delete -I'
    if module.check_mode:
        remove_cmd += 'n'
    if module.params['clean']:
        remove_cmd += 'c'
    if module.params['quick']:
        remove_cmd += 'q'
    for name in names:
        if pkg_spec[name]['installed_state'] is True:
            pkg_spec[name]['rc'], pkg_spec[name]['stdout'], pkg_spec[name]['stderr'] = execute_command('%s %s' % (remove_cmd, name), module)
            if pkg_spec[name]['rc'] == 0:
                pkg_spec[name]['changed'] = True
            else:
                pkg_spec[name]['changed'] = False
        else:
            pkg_spec[name]['rc'] = 0
            pkg_spec[name]['stdout'] = ''
            pkg_spec[name]['stderr'] = ''
            pkg_spec[name]['changed'] = False