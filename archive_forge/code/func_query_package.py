from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def query_package(module, xbps_path, name, state='present'):
    """Returns Package info"""
    if state == 'present':
        lcmd = '%s %s' % (xbps_path['query'], name)
        lrc, lstdout, lstderr = module.run_command(lcmd, check_rc=False)
        if not is_installed(lstdout):
            return (False, False)
        rcmd = '%s -Sun' % xbps_path['install']
        rrc, rstdout, rstderr = module.run_command(rcmd, check_rc=False)
        if rrc == 0 or rrc == 17:
            'Return True to indicate that the package is installed locally,\n            and the result of the version number comparison to determine if the\n            package is up-to-date'
            return (True, name not in rstdout)
        return (False, False)