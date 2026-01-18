from __future__ import absolute_import, division, print_function
import os
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native, to_text, to_bytes
def embedded_wait_for_upgrade(self):
    """Wait for SANtricity Web Services Embedded to be available after reboot."""
    for count in range(0, self.REBOOT_TIMEOUT_SEC):
        try:
            rc, response = self.request('storage-systems/%s/graph/xpath-filter?query=/sa/saData' % self.ssid)
            bundle_display = [m['versionString'] for m in response[0]['extendedSAData']['codeVersions'] if m['codeModule'] == 'bundleDisplay'][0]
            if rc == 200 and six.b(bundle_display) == self.firmware_version() and (six.b(response[0]['nvsramVersion']) == self.nvsram_version()):
                self.upgrade_in_progress = False
                break
        except Exception as error:
            pass
        sleep(1)
    else:
        self.module.fail_json(msg='Timeout waiting for Santricity Web Services Embedded. Array [%s]' % self.ssid)