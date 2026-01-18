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
def get_all_regular_files(basepath):
    for fname in os.listdir(basepath):
        path = os.path.join(basepath, fname)
        if os.path.isfile(path):
            yield path