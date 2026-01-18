from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def remove_monitoring_policy(module, oneandone_conn):
    """
    Removes a monitoring policy.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object
    """
    try:
        mp_id = module.params.get('name')
        monitoring_policy_id = get_monitoring_policy(oneandone_conn, mp_id)
        if module.check_mode:
            if monitoring_policy_id is None:
                _check_mode(module, False)
            _check_mode(module, True)
        monitoring_policy = oneandone_conn.delete_monitoring_policy(monitoring_policy_id)
        changed = True if monitoring_policy else False
        return (changed, {'id': monitoring_policy['id'], 'name': monitoring_policy['name']})
    except Exception as ex:
        module.fail_json(msg=str(ex))