from contextlib import contextmanager
import errno
import os
import stat
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import fileutils
import xattr
from glance.common import exception
from glance.i18n import _, _LI
from glance.image_cache.drivers import base
def set_xattr(path, key, value):
    """Set the value of a specified xattr.

    If xattrs aren't supported by the file-system, we skip setting the value.
    """
    namespaced_key = _make_namespaced_xattr_key(key)
    if not isinstance(value, bytes):
        value = str(value).encode('utf-8')
    xattr.setxattr(path, namespaced_key, value)