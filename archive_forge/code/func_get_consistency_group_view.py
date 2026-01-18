from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_consistency_group_view(self):
    """Determine and return consistency group view."""
    group_id = self.get_consistency_group()['consistency_group_id']
    if not self.cache['get_consistency_group_view']:
        try:
            rc, views = self.request('storage-systems/%s/consistency-groups/%s/views' % (self.ssid, group_id))
            for view in views:
                if view['name'] == self.view_name:
                    self.cache['get_consistency_group_view'] = view
                    self.cache['get_consistency_group_view'].update({'snapshot_volumes': []})
                    try:
                        rc, snapshot_volumes = self.request('storage-systems/%s/snapshot-volumes' % self.ssid)
                        for snapshot_volume in snapshot_volumes:
                            if snapshot_volume['membership'] and snapshot_volume['membership']['viewType'] == 'member' and (snapshot_volume['membership']['cgViewRef'] == view['cgViewRef']):
                                self.cache['get_consistency_group_view']['snapshot_volumes'].append(snapshot_volume)
                    except Exception as error:
                        self.module.fail_json(msg='Failed to retrieve host mapping information!. Group [%s]. Array [%s]. Error [%s].' % (self.group_name, self.ssid, error))
        except Exception as error:
            self.module.fail_json(msg="Failed to retrieve consistency group's views! Group [%s]. Array [%s]. Error [%s]." % (self.group_name, self.ssid, error))
    return self.cache['get_consistency_group_view']