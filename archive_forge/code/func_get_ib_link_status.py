from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_ib_link_status(self):
    """Determine the infiniband link status. Returns dictionary keyed by interface reference number."""
    link_statuses = {}
    try:
        rc, result = self.request('storage-systems/%s/hardware-inventory' % self.ssid)
        for link in result['ibPorts']:
            link_statuses.update({link['channelPortRef']: link['linkState']})
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve ib link status information! Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
    return link_statuses