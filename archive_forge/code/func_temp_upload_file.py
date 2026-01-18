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
@property
def temp_upload_file(self):
    if self._values['temp_upload_file'] is None:
        f = tempfile.NamedTemporaryFile()
        name = os.path.basename(f.name)
        self._values['temp_upload_file'] = name
    return self._values['temp_upload_file']