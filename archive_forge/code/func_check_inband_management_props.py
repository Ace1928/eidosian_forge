from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_inband_management_props(ucs, module, dn):
    props_match = False
    child_dn = dn + '/iface-in-band'
    mo_1 = ucs.login_handle.query_dn(child_dn)
    if mo_1:
        kwargs = dict(mode=module.params['mgmt_interface_mode'])
        if mo_1.check_prop_match(**kwargs):
            child_dn = child_dn + '/network'
            mo_2 = ucs.login_handle.query_dn(child_dn)
            if mo_2:
                kwargs = dict(name=module.params['mgmt_vnet_name'])
                if mo_2.check_prop_match(**kwargs):
                    child_dn = child_dn + '/ipv4-pooled-addr'
                    mo_3 = ucs.login_handle.query_dn(child_dn)
                    if mo_3:
                        kwargs = dict(name=module.params['mgmt_inband_pool_name'])
                        if mo_3.check_prop_match(**kwargs):
                            props_match = True
    elif not module.params['mgmt_interface_mode']:
        props_match = True
    return props_match