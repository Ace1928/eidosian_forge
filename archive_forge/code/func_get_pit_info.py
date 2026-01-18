from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_pit_info(self):
    """Determine consistency group's snapshot images base on provided arguments (pit_name or timestamp)."""

    def _check_timestamp(timestamp):
        """Check whether timestamp matches I(pit_timestamp)"""
        return self.pit_timestamp.year == timestamp.year and self.pit_timestamp.month == timestamp.month and (self.pit_timestamp.day == timestamp.day) and (self.pit_timestamp_tokens < 4 or self.pit_timestamp.hour == timestamp.hour) and (self.pit_timestamp_tokens < 5 or self.pit_timestamp.minute == timestamp.minute) and (self.pit_timestamp_tokens < 6 or self.pit_timestamp.second == timestamp.second)
    if self.cache['get_pit_info'] is None:
        group = self.get_consistency_group()
        pit_images_by_timestamp = self.get_pit_images_by_timestamp()
        pit_images_by_name = self.get_pit_images_by_name()
        if self.pit_name:
            if self.pit_name in pit_images_by_name.keys():
                self.cache['get_pit_info'] = pit_images_by_name[self.pit_name]
                if self.pit_timestamp:
                    for image in self.cache['get_pit_info']['images']:
                        if not _check_timestamp(image['timestamp']):
                            self.module.fail_json(msg='Snapshot image does not exist that matches both name and supplied timestamp! Group [%s]. Image [%s]. Array [%s].' % (self.group_name, image, self.ssid))
        elif self.pit_timestamp and pit_images_by_timestamp:
            sequence_number = None
            if self.pit_timestamp == 'newest':
                sequence_number = group['sequence_numbers'][-1]
                for image_timestamp in pit_images_by_timestamp.keys():
                    if int(pit_images_by_timestamp[image_timestamp]['sequence_number']) == int(sequence_number):
                        self.cache['get_pit_info'] = pit_images_by_timestamp[image_timestamp]
                        break
            elif self.pit_timestamp == 'oldest':
                sequence_number = group['sequence_numbers'][0]
                for image_timestamp in pit_images_by_timestamp.keys():
                    if int(pit_images_by_timestamp[image_timestamp]['sequence_number']) == int(sequence_number):
                        self.cache['get_pit_info'] = pit_images_by_timestamp[image_timestamp]
                        break
            else:
                for image_timestamp in pit_images_by_timestamp.keys():
                    if _check_timestamp(image_timestamp):
                        if sequence_number and sequence_number != pit_images_by_timestamp[image_timestamp]['sequence_number']:
                            self.module.fail_json(msg='Multiple snapshot images match the provided timestamp and do not have the same sequence number! Group [%s]. Array [%s].' % (self.group_name, self.ssid))
                        sequence_number = pit_images_by_timestamp[image_timestamp]['sequence_number']
                        self.cache['get_pit_info'] = pit_images_by_timestamp[image_timestamp]
    if self.state != 'absent' and self.type != 'pit' and (self.cache['get_pit_info'] is None):
        self.module.fail_json(msg='Snapshot consistency group point-in-time image does not exist! Name [%s]. Timestamp [%s]. Group [%s]. Array [%s].' % (self.pit_name, self.pit_timestamp, self.group_name, self.ssid))
    return self.cache['get_pit_info']