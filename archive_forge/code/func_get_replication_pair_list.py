from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_replication_pair_list(self, filter_dict=None):
    """ Get the list of replication pairs on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting replication pair list ')
        if filter_dict:
            pairs = self.powerflex_conn.replication_pair.get(filter_fields=filter_dict)
        else:
            pairs = self.powerflex_conn.replication_pair.get()
        if pairs:
            for pair in pairs:
                pair.pop('links', None)
                local_volume = self.powerflex_conn.volume.get(filter_fields={'id': pair['localVolumeId']})
                if local_volume:
                    pair['localVolumeName'] = local_volume[0]['name']
                pair['replicationConsistencyGroupName'] = self.powerflex_conn.replication_consistency_group.get(filter_fields={'id': pair['replicationConsistencyGroupId']})[0]['name']
                pair['statistics'] = self.powerflex_conn.replication_pair.get_statistics(pair['id'])
            return pairs
    except Exception as e:
        msg = 'Get replication pair list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)