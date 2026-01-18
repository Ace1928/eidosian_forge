from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_pd_list(self, filter_dict=None):
    """ Get the list of Protection Domains on a given PowerFlex
            storage system """
    try:
        LOG.info('Getting protection domain list ')
        if filter_dict:
            pd = self.powerflex_conn.protection_domain.get(filter_fields=filter_dict)
        else:
            pd = self.powerflex_conn.protection_domain.get()
        return result_list(pd)
    except Exception as e:
        msg = 'Get protection domain list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)