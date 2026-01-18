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
def adios(p):
    """Appropriately delete directory, file or link."""
    if os.path.exists(p) and (not os.path.islink(p)) and os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.unlink(p)