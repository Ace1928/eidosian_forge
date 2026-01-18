from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def firmware_version(self):
    """Retrieve firmware version of the firmware file. Return: bytes string"""
    if self.firmware_version_cache is None:
        with open(self.firmware, 'rb') as fh:
            line = fh.readline()
            while line:
                if self.is_firmware_bundled():
                    if b'displayableAttributeList=' in line:
                        for item in line[25:].split(b','):
                            key, value = item.split(b'|')
                            if key == b'VERSION':
                                self.firmware_version_cache = value.strip(b'\n')
                        break
                elif b'Version:' in line:
                    self.firmware_version_cache = line.split()[-1].strip(b'\n')
                    break
                line = fh.readline()
            else:
                self.module.fail_json(msg='Failed to determine firmware version. File [%s]. Array [%s].' % (self.firmware, self.ssid))
    return self.firmware_version_cache