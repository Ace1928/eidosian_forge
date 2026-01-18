from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_storage_pool_list(self, filter_dict=None):
    """ Get the list of storage pools on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting storage pool list ')
        if filter_dict:
            pool = self.powerflex_conn.storage_pool.get(filter_fields=filter_dict)
        else:
            pool = self.powerflex_conn.storage_pool.get()
        if pool:
            statistics_map = self.powerflex_conn.utility.get_statistics_for_all_storagepools()
            list_of_pool_ids_in_statistics = statistics_map.keys()
            for item in pool:
                item['statistics'] = statistics_map[item['id']] if item['id'] in list_of_pool_ids_in_statistics else {}
        return result_list(pool)
    except Exception as e:
        msg = 'Get storage pool list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)