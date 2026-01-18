from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def reduce_vg(module, vg, pvs, vg_validation):
    vg_state, msg = vg_validation
    if vg_state is False:
        changed = False
        return (changed, msg)
    elif vg_state is None:
        changed = False
        return (changed, msg)
    if pvs is None:
        lsvg_cmd = module.get_bin_path('lsvg', True)
        rc, current_pvs, err = module.run_command([lsvg_cmd, '-p', vg])
        if rc != 0:
            module.fail_json(msg="Failing to execute '%s' command." % lsvg_cmd)
        pvs_to_remove = []
        for line in current_pvs.splitlines()[2:]:
            pvs_to_remove.append(line.split()[0])
        reduce_msg = "Volume group '%s' removed." % vg
    else:
        pvs_to_remove = pvs
        reduce_msg = "Physical volume(s) '%s' removed from Volume group '%s'." % (' '.join(pvs_to_remove), vg)
    if len(pvs_to_remove) <= 0:
        changed = False
        msg = 'No physical volumes to remove.'
        return (changed, msg)
    changed = True
    msg = ''
    if not module.check_mode:
        reducevg_cmd = module.get_bin_path('reducevg', True)
        rc, stdout, stderr = module.run_command([reducevg_cmd, '-df', vg] + pvs_to_remove)
        if rc != 0:
            module.fail_json(msg="Unable to remove '%s'." % vg, rc=rc, stdout=stdout, stderr=stderr)
    msg = reduce_msg
    return (changed, msg)