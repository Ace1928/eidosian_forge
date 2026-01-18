from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def is_root_username_credential(self):
    user = self.module.params.get('username_credential', None)
    if user == 'root':
        return True
    return False