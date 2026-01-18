from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def serialize_pnics(vswitch_obj):
    """Get pnic names"""
    pnics = []
    for pnic in vswitch_obj.pnic:
        pnics.append(pnic.split('-', 3)[-1])
    return pnics