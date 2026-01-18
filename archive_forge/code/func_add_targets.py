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
def add_targets(self):
    self.open()
    try:
        for target in self.targets:
            if os.path.isdir(target):
                for directory_path, directory_names, file_names in os.walk(target, topdown=True):
                    for directory_name in directory_names:
                        full_path = os.path.join(directory_path, directory_name)
                        self.add(full_path, strip_prefix(self.root, full_path))
                    for file_name in file_names:
                        full_path = os.path.join(directory_path, file_name)
                        self.add(full_path, strip_prefix(self.root, full_path))
            else:
                self.add(target, strip_prefix(self.root, target))
    except Exception as e:
        if self.format in ('zip', 'tar'):
            archive_format = self.format
        else:
            archive_format = 'tar.' + self.format
        self.module.fail_json(msg='Error when writing %s archive at %s: %s' % (archive_format, _to_native(self.destination), _to_native(e)), exception=format_exc())
    self.close()
    if self.errors:
        self.module.fail_json(msg='Errors when writing archive at %s: %s' % (_to_native(self.destination), '; '.join(self.errors)))