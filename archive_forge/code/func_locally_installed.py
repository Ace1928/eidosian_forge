from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
def locally_installed(module, pkgname):
    rc, out, err = module.run_command('{0} -q {1}'.format(module.get_bin_path('rpm'), pkgname).split())
    return rc == 0