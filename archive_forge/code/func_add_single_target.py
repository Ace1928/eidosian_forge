from __future__ import absolute_import, division, print_function
import abc
import bz2
import glob
import gzip
import io
import os
import re
import shutil
import tarfile
import zipfile
from fnmatch import fnmatch
from sys import version_info
from traceback import format_exc
from zlib import crc32
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils import six
def add_single_target(self, path):
    if self.format in ('zip', 'tar'):
        self.open()
        self.add(path, strip_prefix(self.root, path))
        self.close()
        self.destination_state = STATE_ARCHIVED
    else:
        try:
            f_out = self._open_compressed_file(_to_native_ascii(self.destination), 'wb')
            with open(path, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
            f_out.close()
            self.successes.append(path)
            self.destination_state = STATE_COMPRESSED
        except (IOError, OSError) as e:
            self.module.fail_json(path=_to_native(path), dest=_to_native(self.destination), msg='Unable to write to compressed file: %s' % _to_native(e), exception=format_exc())