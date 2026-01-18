from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_storage_profile_props(ucs, module, dn):
    props_match = False
    child_dn = dn + '/profile-binding'
    mo_1 = ucs.login_handle.query_dn(child_dn)
    if mo_1:
        kwargs = dict(storage_profile_name=module.params['storage_profile'])
        if mo_1.check_prop_match(**kwargs):
            props_match = True
    elif not module.params['storage_profile']:
        props_match = True
    return props_match