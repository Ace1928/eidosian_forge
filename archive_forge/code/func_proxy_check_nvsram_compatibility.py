from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def proxy_check_nvsram_compatibility(self, retries=10):
    """Verify nvsram is compatible with E-Series storage system."""
    self.module.log('Checking nvsram compatibility...')
    data = {'storageDeviceIds': [self.ssid]}
    try:
        rc, check = self.request('firmware/compatibility-check', method='POST', data=data)
    except Exception as error:
        if retries:
            sleep(1)
            self.proxy_check_nvsram_compatibility(retries - 1)
        else:
            self.module.fail_json(msg='Failed to receive NVSRAM compatibility information. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    for count in range(int(self.COMPATIBILITY_CHECK_TIMEOUT_SEC / 5)):
        try:
            rc, response = self.request('firmware/compatibility-check?requestId=%s' % check['requestId'])
        except Exception as error:
            continue
        if not response['checkRunning']:
            for result in response['results'][0]['nvsramFiles']:
                if result['filename'] == self.nvsram_name:
                    return
            self.module.fail_json(msg='NVSRAM is not compatible. NVSRAM [%s]. Array [%s].' % (self.nvsram_name, self.ssid))
        sleep(5)
    self.module.fail_json(msg='Failed to retrieve NVSRAM status update from proxy. Array [%s].' % self.ssid)