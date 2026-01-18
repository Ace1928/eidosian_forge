from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_pit_images_metadata(self):
    """Retrieve and return consistency group snapshot images' metadata keyed on timestamps."""
    if not self.cache['get_pit_images_metadata']:
        try:
            rc, key_values = self.request(self.url_path_prefix + 'key-values')
            for entry in key_values:
                if re.search('ansible\\|%s\\|' % self.group_name, entry['key']):
                    name = entry['key'].replace('ansible|%s|' % self.group_name, '')
                    values = entry['value'].split('|')
                    if len(values) == 3:
                        timestamp, image_id, description = values
                        self.cache['get_pit_images_metadata'].update({timestamp: {'name': name, 'description': description}})
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve consistency group snapshot images metadata!  Array [%s]. Error [%s].' % (self.ssid, error))
    return self.cache['get_pit_images_metadata']