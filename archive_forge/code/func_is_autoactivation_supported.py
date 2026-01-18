from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def is_autoactivation_supported(module, vg_cmd):
    autoactivation_supported = False
    dummy, vgchange_opts, dummy = module.run_command([vg_cmd, '--help'], check_rc=True)
    if VG_AUTOACTIVATION_OPT in vgchange_opts:
        autoactivation_supported = True
    return autoactivation_supported