from __future__ import (absolute_import, division, print_function)
import os
import re
import tempfile
from ansible.module_utils.six import PY2
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
def current_type(self):
    magic_bytes = b'\xfe\xed\xfe\xed'
    with open(self.keystore_path, 'rb') as fd:
        header = fd.read(4)
    if header == magic_bytes:
        return 'jks'
    return 'pkcs12'