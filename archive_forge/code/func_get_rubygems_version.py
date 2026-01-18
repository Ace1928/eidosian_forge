from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_rubygems_version(module):
    if hasattr(get_rubygems_version, 'ver'):
        return get_rubygems_version.ver
    cmd = get_rubygems_path(module) + ['--version']
    rc, out, err = module.run_command(cmd, check_rc=True)
    match = re.match('^(\\d+)\\.(\\d+)\\.(\\d+)', out)
    if not match:
        return None
    ver = tuple((int(x) for x in match.groups()))
    get_rubygems_version.ver = ver
    return ver