from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_disk_policy_props(ucs, module, mo, dn):
    props_match = True
    kwargs = dict(descr=module.params['description'])
    kwargs['raid_level'] = module.params['raid_level']
    if mo.check_prop_match(**kwargs):
        if module.params['configuration_mode'] == 'automatic':
            child_dn = dn + '/disk-group-qual'
            mo_1 = ucs.login_handle.query_dn(child_dn)
            if mo_1:
                kwargs = dict(num_drives=module.params['num_drives'])
                kwargs['drive_type'] = module.params['drive_type']
                kwargs['use_remaining_disks'] = module.params['use_remaining_disks']
                kwargs['num_ded_hot_spares'] = module.params['num_ded_hot_spares']
                kwargs['num_glob_hot_spares'] = module.params['num_glob_hot_spares']
                kwargs['min_drive_size'] = module.params['min_drive_size']
                props_match = mo_1.check_prop_match(**kwargs)
        else:
            for disk in module.params['manual_disks']:
                child_dn = dn + '/slot-' + disk['slot_num']
                mo_1 = ucs.login_handle.query_dn(child_dn)
                if mo_1:
                    if disk['state'] == 'absent':
                        props_match = False
                    else:
                        kwargs = dict(slot_num=disk['slot_num'])
                        kwargs['role'] = disk['role']
                        kwargs['span_id'] = disk['span_id']
                        if not mo_1.check_prop_match(**kwargs):
                            props_match = False
                            break
        if props_match:
            if module.params['virtual_drive']:
                props_match = check_virtual_drive_props(ucs, module, dn)
    else:
        props_match = False
    return props_match