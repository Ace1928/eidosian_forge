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
def delete_cached_image(self, image_id):
    """
        Removes a specific cached image file and any attributes about the image

        :param image_id: Image ID
        """
    path = self.get_image_filepath(image_id)
    delete_cached_file(path)