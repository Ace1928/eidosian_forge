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
def reap_invalid(self, grace=None):
    """Remove any invalid cache entries

        :param grace: Number of seconds to keep an invalid entry around for
                      debugging purposes. If None, then delete immediately.
        """
    return self._reap_old_files(self.invalid_dir, 'invalid', grace=grace)