from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def proxy_upload_and_check_compatibility(self):
    """Ensure firmware/nvsram file is uploaded and verify compatibility."""
    uploaded_files = []
    try:
        rc, uploaded_files = self.request('firmware/cfw-files')
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve uploaded firmware and nvsram files. Error [%s]' % to_native(error))
    if self.firmware:
        for uploaded_file in uploaded_files:
            if uploaded_file['filename'] == self.firmware_name:
                break
        else:
            fields = [('validate', 'true')]
            files = [('firmwareFile', self.firmware_name, self.firmware)]
            headers, data = create_multipart_formdata(files=files, fields=fields)
            try:
                rc, response = self.request('firmware/upload', method='POST', data=data, headers=headers)
            except Exception as error:
                self.module.fail_json(msg='Failed to upload firmware bundle file. File [%s]. Array [%s]. Error [%s].' % (self.firmware_name, self.ssid, to_native(error)))
        self.proxy_check_firmware_compatibility()
    if self.nvsram:
        for uploaded_file in uploaded_files:
            if uploaded_file['filename'] == self.nvsram_name:
                break
        else:
            fields = [('validate', 'true')]
            files = [('firmwareFile', self.nvsram_name, self.nvsram)]
            headers, data = create_multipart_formdata(files=files, fields=fields)
            try:
                rc, response = self.request('firmware/upload', method='POST', data=data, headers=headers)
            except Exception as error:
                self.module.fail_json(msg='Failed to upload NVSRAM file. File [%s]. Array [%s]. Error [%s].' % (self.nvsram_name, self.ssid, to_native(error)))
        self.proxy_check_nvsram_compatibility()