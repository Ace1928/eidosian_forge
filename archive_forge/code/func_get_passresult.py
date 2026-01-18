from __future__ import (absolute_import, division, print_function)
from contextlib import contextmanager
import os
import re
import subprocess
import time
import yaml
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
from ansible.utils.encrypt import random_password
from ansible.plugins.lookup import LookupBase
from ansible import constants as C
from ansible_collections.community.general.plugins.module_utils._filelock import FileLock
def get_passresult(self):
    if self.paramvals['returnall']:
        return os.linesep.join(self.passoutput)
    if self.paramvals['subkey'] == 'password':
        return self.password
    elif self.paramvals['subkey'] in self.passdict:
        return self.passdict[self.paramvals['subkey']]
    else:
        return None