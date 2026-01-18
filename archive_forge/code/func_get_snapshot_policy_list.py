from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_snapshot_policy_list(self, filter_dict=None):
    """ Get the list of snapshot schedules on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting snapshot policies list ')
        if filter_dict:
            snapshot_policies = self.powerflex_conn.snapshot_policy.get(filter_fields=filter_dict)
        else:
            snapshot_policies = self.powerflex_conn.snapshot_policy.get()
        if snapshot_policies:
            statistics_map = self.powerflex_conn.utility.get_statistics_for_all_snapshot_policies()
            list_of_snap_pol_ids_in_statistics = statistics_map.keys()
            for item in snapshot_policies:
                item['statistics'] = statistics_map[item['id']] if item['id'] in list_of_snap_pol_ids_in_statistics else {}
        return result_list(snapshot_policies)
    except Exception as e:
        msg = 'Get snapshot policies list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)