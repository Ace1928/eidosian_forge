from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def remove_system(self, ssid):
    """Remove storage system."""
    try:
        rc, storage_system = self.request('storage-systems/%s' % ssid, method='DELETE')
    except Exception as error:
        self.module.warn('Failed to remove storage system. Array [%s]. Error [%s].' % (ssid, to_native(error)))