from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_sds_list(self, filter_dict=None):
    """ Get the list of sdses on a given PowerFlex storage system """
    try:
        LOG.info('Getting SDS list ')
        if filter_dict:
            sds = self.powerflex_conn.sds.get(filter_fields=filter_dict)
        else:
            sds = self.powerflex_conn.sds.get()
        return result_list(sds)
    except Exception as e:
        msg = 'Get SDS list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)