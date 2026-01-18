from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def delete_mel_events(self):
    """Clear all mel-events."""
    try:
        rc, response = self.request('storage-systems/%s/mel-events?clearCache=true&resetMel=true' % self.ssid, method='DELETE')
    except Exception as error:
        self.module.fail_json(msg='Failed to clear mel-events. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))