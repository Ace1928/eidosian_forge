from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def configure_disk_policy(ucs, module, dn):
    from ucsmsdk.mometa.lstorage.LstorageDiskGroupConfigPolicy import LstorageDiskGroupConfigPolicy
    from ucsmsdk.mometa.lstorage.LstorageDiskGroupQualifier import LstorageDiskGroupQualifier
    from ucsmsdk.mometa.lstorage.LstorageLocalDiskConfigRef import LstorageLocalDiskConfigRef
    if not module.check_mode:
        try:
            mo = LstorageDiskGroupConfigPolicy(parent_mo_or_dn=module.params['org_dn'], name=module.params['name'], descr=module.params['description'], raid_level=module.params['raid_level'])
            if module.params['configuration_mode'] == 'automatic':
                LstorageDiskGroupQualifier(parent_mo_or_dn=mo, num_drives=module.params['num_drives'], drive_type=module.params['drive_type'], use_remaining_disks=module.params['use_remaining_disks'], num_ded_hot_spares=module.params['num_ded_hot_spares'], num_glob_hot_spares=module.params['num_glob_hot_spares'], min_drive_size=module.params['min_drive_size'])
            else:
                for disk in module.params['manual_disks']:
                    if disk['state'] == 'absent':
                        child_dn = dn + '/slot-' + disk['slot_num']
                        mo_1 = ucs.login_handle.query_dn(child_dn)
                        if mo_1:
                            ucs.login_handle.remove_mo(mo_1)
                    else:
                        LstorageLocalDiskConfigRef(parent_mo_or_dn=mo, slot_num=disk['slot_num'], role=disk['role'], span_id=disk['span_id'])
            if module.params['virtual_drive']:
                _configure_virtual_drive(module, mo)
            ucs.login_handle.add_mo(mo, True)
            ucs.login_handle.commit()
        except Exception as e:
            ucs.result['msg'] = 'setup error: %s ' % str(e)
            module.fail_json(**ucs.result)
    ucs.result['changed'] = True