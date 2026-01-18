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
@contextmanager
def opt_lock(self, type):
    if self.get_option('lock') == type:
        tmpdir = os.environ.get('TMPDIR', '/tmp')
        lockfile = os.path.join(tmpdir, '.passwordstore.lock')
        with FileLock().lock_file(lockfile, tmpdir, self.lock_timeout):
            self.locked = type
            yield
        self.locked = None
    else:
        yield