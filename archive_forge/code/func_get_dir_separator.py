import sys
import urllib.parse as urlparse
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import glance.db as db_api
from glance.i18n import _LE, _LW
from glance import scrubber
def get_dir_separator():
    separator = ''
    staging_dir = 'file://%s' % getattr(CONF, 'os_glance_staging_store').filesystem_store_datadir
    if not staging_dir.endswith('/'):
        separator = '/'
    return (separator, staging_dir)