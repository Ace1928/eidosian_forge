from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def requires_ontap_9_6(self, module_name):
    return self.requires_ontap_version(module_name)