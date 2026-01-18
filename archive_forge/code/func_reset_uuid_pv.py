from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def reset_uuid_pv(module, device):
    changed = False
    pvs_cmd = module.get_bin_path('pvs', True)
    pvs_cmd_with_opts = [pvs_cmd, '--noheadings', '-o', 'uuid', device]
    pvchange_cmd = module.get_bin_path('pvchange', True)
    pvchange_cmd_with_opts = [pvchange_cmd, '-u', device]
    dummy, orig_uuid, dummy = module.run_command(pvs_cmd_with_opts, check_rc=True)
    if module.check_mode:
        changed = True
    else:
        pvchange_rc, pvchange_out, pvchange_err = module.run_command(pvchange_cmd_with_opts)
        dummy, new_uuid, dummy = module.run_command(pvs_cmd_with_opts, check_rc=True)
        if orig_uuid.strip() == new_uuid.strip():
            module.fail_json(msg='PV (%s) UUID change failed' % device, rc=pvchange_rc, err=pvchange_err, out=pvchange_out)
        else:
            changed = True
    return changed