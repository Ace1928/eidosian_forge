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
def encrypt_file(self, filename, secret, vault_id=None, output_file=None):
    filename = self._real_path(filename)
    b_plaintext = self.read_data(filename)
    b_ciphertext = self.vault.encrypt(b_plaintext, secret, vault_id=vault_id)
    self.write_data(b_ciphertext, output_file or filename)