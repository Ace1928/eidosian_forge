from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def embedded_check_nvsram_compatibility(self):
    """Verify the provided NVSRAM is compatible with E-Series storage system."""
    files = [('nvsramimage', self.nvsram_name, self.nvsram)]
    headers, data = create_multipart_formdata(files=files)
    compatible = {}
    try:
        rc, compatible = self.request('firmware/embedded-firmware/%s/nvsram-compatibility-check' % self.ssid, method='POST', data=data, headers=headers)
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve NVSRAM compatibility results. Array Id [%s]. Error[%s].' % (self.ssid, to_native(error)))
    if not compatible['signatureTestingPassed']:
        self.module.fail_json(msg='Invalid NVSRAM file. File [%s].' % self.nvsram)
    if not compatible['fileCompatible']:
        self.module.fail_json(msg='Incompatible NVSRAM file. File [%s].' % self.nvsram)
    for module in compatible['versionContents']:
        if module['bundledVersion'] != module['onboardVersion']:
            self.upgrade_required = True
        self.module_info.update({module['module']: {'onboard_version': module['onboardVersion'], 'bundled_version': module['bundledVersion']}})