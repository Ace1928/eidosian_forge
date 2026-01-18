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
def get_least_recently_accessed(self):
    """
        Return a tuple containing the image_id and size of the least recently
        accessed cached file, or None if no cached files.
        """
    stats = []
    for path in get_all_regular_files(self.base_dir):
        file_info = os.stat(path)
        stats.append((file_info[stat.ST_ATIME], file_info[stat.ST_SIZE], path))
    if not stats:
        return None
    stats.sort()
    return (os.path.basename(stats[0][2]), stats[0][1])