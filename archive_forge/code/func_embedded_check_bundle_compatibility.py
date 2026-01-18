from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def embedded_check_bundle_compatibility(self):
    """Verify the provided firmware bundle is compatible with E-Series storage system."""
    files = [('files[]', 'blob', self.firmware)]
    headers, data = create_multipart_formdata(files=files, send_8kb=True)
    compatible = {}
    try:
        rc, compatible = self.request('firmware/embedded-firmware/%s/bundle-compatibility-check' % self.ssid, method='POST', data=data, headers=headers)
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve bundle compatibility results. Array Id [%s]. Error[%s].' % (self.ssid, to_native(error)))
    if not compatible['signatureTestingPassed']:
        self.module.fail_json(msg='Invalid firmware bundle file. File [%s].' % self.firmware)
    if not compatible['fileCompatible']:
        self.module.fail_json(msg='Incompatible firmware bundle file. File [%s].' % self.firmware)
    for module in compatible['versionContents']:
        bundle_module_version = module['bundledVersion'].split('.')
        onboard_module_version = module['onboardVersion'].split('.')
        version_minimum_length = min(len(bundle_module_version), len(onboard_module_version))
        if bundle_module_version[:version_minimum_length] != onboard_module_version[:version_minimum_length]:
            self.upgrade_required = True
        self.module_info.update({module['module']: {'onboard_version': module['onboardVersion'], 'bundled_version': module['bundledVersion']}})