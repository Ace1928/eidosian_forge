from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_lan_connecivity_props(ucs, module, mo, dn):
    props_match = False
    kwargs = dict(descr=module.params['description'])
    if mo.check_prop_match(**kwargs):
        props_match = check_vnic_props(ucs, module, dn)
        if props_match:
            props_match = check_iscsi_vnic_props(ucs, module, dn)
    return props_match