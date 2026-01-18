from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def configure_lan_connectivity(ucs, module, dn):
    from ucsmsdk.mometa.vnic.VnicLanConnPolicy import VnicLanConnPolicy
    from ucsmsdk.mometa.vnic.VnicEther import VnicEther
    from ucsmsdk.mometa.vnic.VnicIScsiLCP import VnicIScsiLCP
    from ucsmsdk.mometa.vnic.VnicVlan import VnicVlan
    if not module.check_mode:
        try:
            mo = VnicLanConnPolicy(parent_mo_or_dn=module.params['org_dn'], name=module.params['name'], descr=module.params['description'])
            if module.params.get('vnic_list'):
                for vnic in module.params['vnic_list']:
                    if vnic['state'] == 'absent':
                        child_dn = dn + '/ether-' + vnic['name']
                        mo_1 = ucs.login_handle.query_dn(child_dn)
                        if mo_1:
                            ucs.login_handle.remove_mo(mo_1)
                    else:
                        mo_1 = VnicEther(addr='derived', parent_mo_or_dn=mo, name=vnic['name'], adaptor_profile_name=vnic['adapter_policy'], nw_templ_name=vnic['vnic_template'], order=vnic['order'])
            if module.params.get('iscsi_vnic_list'):
                for iscsi_vnic in module.params['iscsi_vnic_list']:
                    if iscsi_vnic['state'] == 'absent':
                        child_dn = dn + '/iscsi-' + iscsi_vnic['name']
                        mo_1 = ucs.login_handle.query_dn(child_dn)
                        if mo_1:
                            ucs.login_handle.remove_mo(mo_1)
                    else:
                        mo_1 = VnicIScsiLCP(parent_mo_or_dn=mo, name=iscsi_vnic['name'], adaptor_profile_name=iscsi_vnic['iscsi_adapter_policy'], vnic_name=iscsi_vnic['overlay_vnic'], addr=iscsi_vnic['mac_address'])
                        VnicVlan(parent_mo_or_dn=mo_1, vlan_name=iscsi_vnic['vlan_name'])
            ucs.login_handle.add_mo(mo, True)
            ucs.login_handle.commit()
        except Exception as e:
            ucs.result['msg'] = 'setup error: %s ' % str(e)
            module.fail_json(**ucs.result)
    ucs.result['changed'] = True