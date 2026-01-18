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
def find_targets(self):
    for path in self.paths:
        if not os.path.lexists(path):
            self.not_found.append(path)
        else:
            self.targets.append(path)