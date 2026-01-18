from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def check_controller(self):
    """Is the effected controller the alternate controller."""
    controllers_info = self.get_controllers()
    try:
        rc, about = self.request('utils/about', rest_api_path=self.DEFAULT_BASE_PATH)
        self.url_path_suffix = '?alternate=%s' % ('true' if controllers_info[self.controller] != about['controllerPosition'] else 'false')
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve accessing controller slot information. Array [%s].' % self.ssid)