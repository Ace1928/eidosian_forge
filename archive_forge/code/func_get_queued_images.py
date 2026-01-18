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
def get_queued_images(self):
    """
        Returns a list of image IDs that are in the queue. The
        list should be sorted by the time the image ID was inserted
        into the queue.
        """
    files = [f for f in get_all_regular_files(self.queue_dir)]
    items = []
    for path in files:
        mtime = os.path.getmtime(path)
        items.append((mtime, os.path.basename(path)))
    items.sort()
    return [image_id for modtime, image_id in items]