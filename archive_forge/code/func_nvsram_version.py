from __future__ import absolute_import, division, print_function
import os
import multiprocessing
import threading
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
def nvsram_version(self):
    """Retrieve NVSRAM version of the NVSRAM file. Return: byte string"""
    if self.nvsram_version_cache is None:
        with open(self.nvsram, 'rb') as fh:
            line = fh.readline()
            while line:
                if b'.NVSRAM Configuration Number' in line:
                    self.nvsram_version_cache = line.split(b'"')[-2]
                    break
                line = fh.readline()
            else:
                self.module.fail_json(msg='Failed to determine NVSRAM file version. File [%s]. Array [%s].' % (self.nvsram, self.ssid))
    return self.nvsram_version_cache