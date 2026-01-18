from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def create_pit_images(self):
    """Generate snapshot image(s) for the base volumes in the consistency group."""
    group_id = self.get_consistency_group()['consistency_group_id']
    try:
        rc, images = self.request('storage-systems/%s/consistency-groups/%s/snapshots' % (self.ssid, group_id), method='POST')
        if self.pit_name:
            try:
                rc, key_values = self.request(self.url_path_prefix + 'key-values/ansible|%s|%s' % (self.group_name, self.pit_name), method='POST', data='%s|%s|%s' % (images[0]['pitTimestamp'], self.pit_name, self.pit_description))
            except Exception as error:
                self.module.fail_json(msg='Failed to create metadata for snapshot images! Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))
    except Exception as error:
        self.module.fail_json(msg='Failed to create consistency group snapshot images! Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))