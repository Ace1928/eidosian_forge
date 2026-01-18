from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def get_ontap_version(self):
    if self.ontap_version['valid']:
        return (self.ontap_version['generation'], self.ontap_version['major'], self.ontap_version['minor'])
    return (-1, -1, -1)