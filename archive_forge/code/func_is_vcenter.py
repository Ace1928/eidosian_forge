from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def is_vcenter(self):
    """
        Check if given hostname is vCenter or ESXi host
        Returns: True if given connection is with vCenter server
                 False if given connection is with ESXi server

        """
    api_type = None
    try:
        api_type = self.content.about.apiType
    except (vmodl.RuntimeFault, vim.fault.VimFault) as exc:
        self.module.fail_json(msg='Failed to get status of vCenter server : %s' % exc.msg)
    if api_type == 'VirtualCenter':
        return True
    elif api_type == 'HostAgent':
        return False