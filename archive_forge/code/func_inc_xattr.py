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
def inc_xattr(path, key, n=1):
    """
    Increment the value of an xattr (assuming it is an integer).

    BEWARE, this code *does* have a RACE CONDITION, since the
    read/update/write sequence is not atomic.

    Since the use-case for this function is collecting stats--not critical--
    the benefits of simple, lock-free code out-weighs the possibility of an
    occasional hit not being counted.
    """
    count = int(get_xattr(path, key))
    count += n
    set_xattr(path, key, str(count))