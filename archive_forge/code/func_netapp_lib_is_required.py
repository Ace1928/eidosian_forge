from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def netapp_lib_is_required():
    return 'Error: the python NetApp-Lib module is required.  Import error: %s' % str(IMPORT_EXCEPTION)