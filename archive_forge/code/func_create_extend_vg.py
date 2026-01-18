from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def create_extend_vg(module, vg, pvs, pp_size, vg_type, force, vg_validation):
    """ Creates or extend a volume group. """
    force_opt = {True: '-f', False: ''}
    vg_opt = {'normal': '', 'big': '-B', 'scalable': '-S'}
    pv_state, msg = _validate_pv(module, vg, pvs)
    if not pv_state:
        changed = False
        return (changed, msg)
    vg_state, msg = vg_validation
    if vg_state is False:
        changed = False
        return (changed, msg)
    elif vg_state is True:
        changed = True
        msg = ''
        if not module.check_mode:
            extendvg_cmd = module.get_bin_path('extendvg', True)
            rc, output, err = module.run_command([extendvg_cmd, vg] + pvs)
            if rc != 0:
                changed = False
                msg = "Extending volume group '%s' has failed." % vg
                return (changed, msg)
        msg = "Volume group '%s' extended." % vg
        return (changed, msg)
    elif vg_state is None:
        changed = True
        msg = ''
        if not module.check_mode:
            mkvg_cmd = module.get_bin_path('mkvg', True)
            rc, output, err = module.run_command([mkvg_cmd, vg_opt[vg_type], pp_size, force_opt[force], '-y', vg] + pvs)
            if rc != 0:
                changed = False
                msg = "Creating volume group '%s' failed." % vg
                return (changed, msg)
        msg = "Volume group '%s' created." % vg
        return (changed, msg)