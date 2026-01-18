from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_vif(module, array, subnet):
    """Delete VLAN Interface"""
    changed = True
    if not module.check_mode:
        vif_name = module.params['name'] + '.' + str(subnet['vlan'])
        try:
            array.delete_vlan_interface(vif_name)
        except Exception:
            module.fail_json(msg='Failed to delete VLAN inerface {0}'.format(vif_name))
    module.exit_json(changed=changed)