from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def remove_firewall_policy(module, oneandone_conn):
    """
    Removes a firewall policy.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object
    """
    try:
        fp_id = module.params.get('name')
        firewall_policy_id = get_firewall_policy(oneandone_conn, fp_id)
        if module.check_mode:
            if firewall_policy_id is None:
                _check_mode(module, False)
            _check_mode(module, True)
        firewall_policy = oneandone_conn.delete_firewall(firewall_policy_id)
        changed = True if firewall_policy else False
        return (changed, {'id': firewall_policy['id'], 'name': firewall_policy['name']})
    except Exception as e:
        module.fail_json(msg=str(e))