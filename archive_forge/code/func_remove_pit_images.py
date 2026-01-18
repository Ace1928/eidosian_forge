from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def remove_pit_images(self, pit_info):
    """Remove selected snapshot point-in-time images."""
    group_id = self.get_consistency_group()['consistency_group_id']
    pit_sequence_number = int(pit_info['sequence_number'])
    sequence_numbers = set((int(pit_image['sequence_number']) for timestamp, pit_image in self.get_pit_images_by_timestamp().items() if int(pit_image['sequence_number']) < pit_sequence_number))
    sequence_numbers.add(pit_sequence_number)
    for sequence_number in sorted(sequence_numbers):
        try:
            rc, images = self.request('storage-systems/%s/consistency-groups/%s/snapshots/%s' % (self.ssid, group_id, sequence_number), method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to create consistency group snapshot images! Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))
    if self.pit_name:
        try:
            rc, key_values = self.request(self.url_path_prefix + 'key-values/ansible|%s|%s' % (self.group_name, self.pit_name), method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to delete metadata for snapshot images! Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))