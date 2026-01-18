from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def options_require_ontap_version(self, options, version='9.6', use_rest=None):
    current_version = self.get_ontap_version()
    suffix = ' - %s' % self.is_rest_error if self.is_rest_error is not None else ''
    if current_version != (-1, -1, -1):
        suffix += ' - ONTAP version: %s.%s.%s' % current_version
    if use_rest is not None:
        suffix += ' - using %s' % ('REST' if use_rest else 'ZAPI')
    if isinstance(options, list) and len(options) > 1:
        tag = 'any of %s' % options
    elif isinstance(options, list) and len(options) == 1:
        tag = str(options[0])
    else:
        tag = str(options)
    return 'using %s requires ONTAP %s or later and REST must be enabled%s.' % (tag, version, suffix)