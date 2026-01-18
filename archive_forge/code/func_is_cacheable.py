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
def is_cacheable(self, image_id):
    """
        Returns True if the image with the supplied ID can have its
        image file cached, False otherwise.

        :param image_id: Image ID
        """
    return not (self.is_cached(image_id) or self.is_being_cached(image_id))