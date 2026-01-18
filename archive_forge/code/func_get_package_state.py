from __future__ import absolute_import, division, print_function
import os
import platform
import re
import shlex
import sqlite3
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_package_state(names, pkg_spec, module):
    info_cmd = 'pkg_info -Iq'
    for name in names:
        command = '%s inst:%s' % (info_cmd, name)
        rc, stdout, stderr = execute_command(command, module)
        if stderr:
            match = re.search("^Can't find inst:%s$" % re.escape(name), stderr)
            if match:
                pkg_spec[name]['installed_state'] = False
            else:
                module.fail_json(msg='failed in get_package_state(): ' + stderr)
        if stdout:
            pkg_spec[name]['installed_names'] = stdout.splitlines()
            module.debug('get_package_state(): installed_names = %s' % pkg_spec[name]['installed_names'])
            pkg_spec[name]['installed_state'] = True
        else:
            pkg_spec[name]['installed_state'] = False