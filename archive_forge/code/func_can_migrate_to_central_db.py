import datetime
import os
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance import context
import glance.db
from glance.i18n import _
from glance.image_cache.drivers import common
def can_migrate_to_central_db():
    if not (CONF.paste_deploy.flavor and 'cache' in CONF.paste_deploy.flavor):
        return False
    is_centralized_db_driver = CONF.image_cache_driver == 'centralized_db'
    if is_centralized_db_driver and (not CONF.worker_self_reference_url):
        msg = _("'worker_self_reference_url' needs to be set if `centralized_db` is defined as cache driver for image_cache_driver config option.")
        raise RuntimeError(msg)
    return is_centralized_db_driver