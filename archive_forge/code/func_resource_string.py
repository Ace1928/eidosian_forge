import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def resource_string(package, resource_name):
    """Load a resource from a package and return it as a string.

    Note: Only packages that start with breezy are currently supported.

    This is designed to be a lightweight implementation of resource
    loading in a way which is API compatible with the same API from
    pkg_resources. See
    http://peak.telecommunity.com/DevCenter/PkgResources#basic-resource-access.
    If and when pkg_resources becomes a standard library, this routine
    can delegate to it.
    """
    if package == 'breezy':
        resource_relpath = resource_name
    elif package.startswith('breezy.'):
        package = package[len('breezy.'):].replace('.', os.sep)
        resource_relpath = pathjoin(package, resource_name)
    else:
        raise errors.BzrError('resource package %s not in breezy' % package)
    base = dirname(breezy.__file__)
    if getattr(sys, 'frozen', None):
        base = abspath(pathjoin(base, '..', '..'))
    with open(pathjoin(base, resource_relpath)) as f:
        return f.read()