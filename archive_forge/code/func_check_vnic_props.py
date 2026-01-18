from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_vnic_props(ucs, module, dn):
    props_match = True
    if module.params.get('vnic_list'):
        for vnic in module.params['vnic_list']:
            child_dn = dn + '/ether-' + vnic['name']
            mo_1 = ucs.login_handle.query_dn(child_dn)
            if mo_1:
                if vnic['state'] == 'absent':
                    props_match = False
                    break
                else:
                    kwargs = dict(adaptor_profile_name=vnic['adapter_policy'])
                    kwargs['order'] = vnic['order']
                    kwargs['nw_templ_name'] = vnic['vnic_template']
                    if not mo_1.check_prop_match(**kwargs):
                        props_match = False
                        break
            elif vnic['state'] == 'present':
                props_match = False
                break
    return props_match