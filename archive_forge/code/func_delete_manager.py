from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_manager(module, blade):
    """Delete SNMP Manager"""
    changed = True
    if not module.check_mode:
        try:
            blade.snmp_managers.delete_snmp_managers(names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Delete SNMP manager {0} failed'.format(module.params['name']))
    module.exit_json(changed=changed)