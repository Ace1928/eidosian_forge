from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_volumes_list(self, filter_dict=None):
    """ Get the list of volumes on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting volumes list ')
        if filter_dict:
            volumes = self.powerflex_conn.volume.get(filter_fields=filter_dict)
        else:
            volumes = self.powerflex_conn.volume.get()
        if volumes:
            statistics_map = self.powerflex_conn.utility.get_statistics_for_all_volumes()
            list_of_vol_ids_in_statistics = statistics_map.keys()
            for item in volumes:
                item['statistics'] = statistics_map[item['id']] if item['id'] in list_of_vol_ids_in_statistics else {}
        return result_list(volumes)
    except Exception as e:
        msg = 'Get volumes list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)