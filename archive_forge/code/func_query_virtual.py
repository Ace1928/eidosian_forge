from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def query_virtual(module, name):
    cmd = '%s -v info --description %s' % (APK_PATH, name)
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    search_pattern = '^%s: virtual meta package' % re.escape(name)
    if re.search(search_pattern, stdout):
        return True
    return False