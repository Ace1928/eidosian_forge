from contextlib import contextmanager
import os
import stat
import time
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import fileutils
from glance.common import exception
from glance import context
import glance.db
from glance.i18n import _LI, _LW
from glance.image_cache.drivers import base
def delete_stalled_files(self, older_than):
    """
        Removes any incomplete cache entries older than a
        supplied modified time.

        :param older_than: Files written to on or before this timestamp
                           will be deleted.
        """
    for path in self.get_cache_files(self.incomplete_dir):
        if os.path.getmtime(path) < older_than:
            try:
                fileutils.delete_if_exists(path)
                LOG.info(_LI('Removed stalled cache file %s'), path)
            except Exception as e:
                msg = (_LW('Failed to delete file %(path)s. Got error: %(e)s'), dict(path=path, e=e))
                LOG.warning(msg)