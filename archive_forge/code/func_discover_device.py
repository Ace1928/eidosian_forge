from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def discover_device(module, device):
    """ Discover AIX devices."""
    cfgmgr_cmd = module.get_bin_path('cfgmgr', True)
    if device is not None:
        device = '-l %s' % device
    else:
        device = ''
    changed = True
    msg = ''
    if not module.check_mode:
        rc, cfgmgr_out, err = module.run_command(['%s' % cfgmgr_cmd, '%s' % device])
        changed = True
        msg = cfgmgr_out
    return (changed, msg)