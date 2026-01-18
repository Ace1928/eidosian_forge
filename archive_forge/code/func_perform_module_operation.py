from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def perform_module_operation(self):
    """
        Perform different actions on replication consistency group based on parameters passed in
        the playbook
        """
    self.validate_input(self.module.params)
    rcg_name = self.module.params['rcg_name']
    new_rcg_name = self.module.params['new_rcg_name']
    rcg_id = self.module.params['rcg_id']
    state = self.module.params['state']
    changed = False
    result = dict(changed=False, replication_consistency_group_details=[])
    rcg_details = self.get_rcg(rcg_name, rcg_id)
    if rcg_details:
        result['replication_consistency_group_details'] = rcg_details
        rcg_id = rcg_details['id']
    msg = 'Fetched the RCG details {0}'.format(str(rcg_details))
    LOG.info(msg)
    if state == 'present':
        if not rcg_details:
            self.validate_create(self.module.params)
            changed, rcg_details = self.create_rcg(self.module.params)
            if rcg_details:
                rcg_id = rcg_details['id']
        if rcg_details and self.modify_rcg(rcg_id, rcg_details):
            changed = True
    if state == 'absent' and rcg_details:
        changed = self.delete_rcg(rcg_id=rcg_details['id'])
    if changed:
        result['replication_consistency_group_details'] = self.get_rcg(new_rcg_name or rcg_name, rcg_id)
    result['changed'] = changed
    self.module.exit_json(**result)