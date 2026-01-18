from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def embedded_firmware_download(self):
    """Execute the firmware download."""
    if self.nvsram:
        firmware_url = 'firmware/embedded-firmware?nvsram=true&staged=true'
        headers, data = create_multipart_formdata(files=[('nvsramfile', self.nvsram_name, self.nvsram), ('dlpfile', self.firmware_name, self.firmware)])
    else:
        firmware_url = 'firmware/embedded-firmware?nvsram=false&staged=true'
        headers, data = create_multipart_formdata(files=[('dlpfile', self.firmware_name, self.firmware)])
    try:
        rc, response = self.request(firmware_url, method='POST', data=data, headers=headers, timeout=30 * 60)
    except Exception as error:
        self.module.fail_json(msg='Failed to stage firmware. Array Id [%s]. Error[%s].' % (self.ssid, to_native(error)))
    activate_thread = threading.Thread(target=self.embedded_firmware_activate)
    activate_thread.start()
    self.wait_for_reboot()