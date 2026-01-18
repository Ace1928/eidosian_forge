from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_virtual_drive_props(ucs, module, dn):
    child_dn = dn + '/virtual-drive-def'
    mo_1 = ucs.login_handle.query_dn(child_dn)
    return mo_1.check_prop_match(**module.params['virtual_drive'])