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
def get_abi_tag():
    """Return the ABI tag based on SOABI (if available) or emulate SOABI (PyPy2)."""
    soabi = sysconfig.get_config_var('SOABI')
    impl = tags.interpreter_name()
    if not soabi and impl in ('cp', 'pp') and hasattr(sys, 'maxunicode'):
        d = ''
        m = ''
        u = ''
        if get_flag('Py_DEBUG', hasattr(sys, 'gettotalrefcount'), warn=impl == 'cp'):
            d = 'd'
        if get_flag('WITH_PYMALLOC', impl == 'cp', warn=impl == 'cp' and sys.version_info < (3, 8)) and sys.version_info < (3, 8):
            m = 'm'
        abi = f'{impl}{tags.interpreter_version()}{d}{m}{u}'
    elif soabi and impl == 'cp' and soabi.startswith('cpython'):
        abi = 'cp' + soabi.split('-')[1]
    elif soabi and impl == 'cp' and soabi.startswith('cp'):
        abi = soabi.split('-')[0]
    elif soabi and impl == 'pp':
        abi = '-'.join(soabi.split('-')[:2])
        abi = abi.replace('.', '_').replace('-', '_')
    elif soabi and impl == 'graalpy':
        abi = '-'.join(soabi.split('-')[:3])
        abi = abi.replace('.', '_').replace('-', '_')
    elif soabi:
        abi = soabi.replace('.', '_').replace('-', '_')
    else:
        abi = None
    return abi