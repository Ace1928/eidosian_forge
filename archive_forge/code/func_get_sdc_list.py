from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_sdc_list(self, filter_dict=None):
    """ Get the list of sdcs on a given PowerFlex storage system """
    try:
        LOG.info('Getting SDC list ')
        if filter_dict:
            sdc = self.powerflex_conn.sdc.get(filter_fields=filter_dict)
        else:
            sdc = self.powerflex_conn.sdc.get()
        return result_list(sdc)
    except Exception as e:
        msg = 'Get SDC list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)