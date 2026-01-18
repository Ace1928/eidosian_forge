from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_interface(module, array):
    changed = True
    if not module.check_mode:
        res = array.delete_network_interfaces(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete network interface {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)