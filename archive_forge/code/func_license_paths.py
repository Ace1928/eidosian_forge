from __future__ import annotations
import os
import re
import shutil
import stat
import struct
import sys
import sysconfig
import warnings
from email.generator import BytesGenerator, Generator
from email.policy import EmailPolicy
from glob import iglob
from shutil import rmtree
from zipfile import ZIP_DEFLATED, ZIP_STORED
import setuptools
from setuptools import Command
from . import __version__ as wheel_version
from .macosx_libfile import calculate_macosx_platform_tag
from .metadata import pkginfo_to_metadata
from .util import log
from .vendored.packaging import tags
from .vendored.packaging import version as _packaging_version
from .wheelfile import WheelFile
@property
def license_paths(self):
    if setuptools_major_version >= 57:
        return self.distribution.metadata.license_files or ()
    files = set()
    metadata = self.distribution.get_option_dict('metadata')
    if setuptools_major_version >= 42:
        patterns = self.distribution.metadata.license_files
    elif 'license_files' in metadata:
        patterns = metadata['license_files'][1].split()
    else:
        patterns = ()
    if 'license_file' in metadata:
        warnings.warn('The "license_file" option is deprecated. Use "license_files" instead.', DeprecationWarning, stacklevel=2)
        files.add(metadata['license_file'][1])
    if not files and (not patterns) and (not isinstance(patterns, list)):
        patterns = ('LICEN[CS]E*', 'COPYING*', 'NOTICE*', 'AUTHORS*')
    for pattern in patterns:
        for path in iglob(pattern):
            if path.endswith('~'):
                log.debug(f'ignoring license file "{path}" as it looks like a backup')
                continue
            if path not in files and os.path.isfile(path):
                log.info(f'adding license file "{path}" (matched pattern "{pattern}")')
                files.add(path)
    return files