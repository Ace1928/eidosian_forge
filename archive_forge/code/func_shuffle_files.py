from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
def shuffle_files(self, src, dest):
    prev = None
    if os.path.isfile(dest):
        prev = os.stat(dest)
        os.remove(dest)
    shutil.move(src, dest)
    if prev is not None:
        os.chmod(dest, prev.st_mode)
        os.chown(dest, prev.st_uid, prev.st_gid)