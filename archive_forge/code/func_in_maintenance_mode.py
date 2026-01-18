from __future__ import absolute_import, division, print_function
import time
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def in_maintenance_mode(self):
    """Determine whether storage device is currently in maintenance mode."""
    results = False
    try:
        rc, key_values = self.request(self.url_path_prefix + 'key-values')
        for key_value in key_values:
            if key_value['key'] == 'ansible_asup_maintenance_email_list':
                if not self.maintenance_emails:
                    self.maintenance_emails = key_value['value'].split(',')
            elif key_value['key'] == 'ansible_asup_maintenance_stop_time':
                if time.time() < float(key_value['value']):
                    results = True
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve maintenance windows information! Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    return results