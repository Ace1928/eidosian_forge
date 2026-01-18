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
def get_cache_size(self):
    """
        Returns the total size in bytes of the image cache.
        """
    sizes = []
    for path in get_all_regular_files(self.base_dir):
        file_info = os.stat(path)
        sizes.append(file_info[stat.ST_SIZE])
    return sum(sizes)