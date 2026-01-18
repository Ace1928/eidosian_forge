from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def format_pkgin_command(module, command, package=None):
    if package is None:
        package = ''
    if module.params['force']:
        force = '-F'
    else:
        force = ''
    vars = {'pkgin': PKGIN_PATH, 'command': command, 'package': package, 'force': force}
    if module.check_mode:
        return '%(pkgin)s -n %(command)s %(package)s' % vars
    else:
        return '%(pkgin)s -y %(force)s %(command)s %(package)s' % vars