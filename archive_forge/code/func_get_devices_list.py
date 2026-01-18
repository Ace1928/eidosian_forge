from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_devices_list(self, filter_dict=None):
    """ Get the list of devices on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting device list ')
        if filter_dict:
            devices = self.powerflex_conn.device.get(filter_fields=filter_dict)
        else:
            devices = self.powerflex_conn.device.get()
        return result_list(devices)
    except Exception as e:
        msg = 'Get device list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)