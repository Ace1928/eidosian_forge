from __future__ import absolute_import, division, print_function
from collections import defaultdict
import re
from ansible.module_utils.basic import AnsibleModule
def autoremove_packages(module, run_pkgng):
    stdout = ''
    stderr = ''
    rc, out, err = run_pkgng('autoremove', '-n')
    autoremove_c = 0
    match = re.search('^Deinstallation has been requested for the following ([0-9]+) packages', out, re.MULTILINE)
    if match:
        autoremove_c = int(match.group(1))
    if autoremove_c == 0:
        return (False, 'no package(s) to autoremove', stdout, stderr)
    if not module.check_mode:
        rc, out, err = run_pkgng('autoremove', '-y')
        stdout += out
        stderr += err
    return (True, 'autoremoved %d package(s)' % autoremove_c, stdout, stderr)