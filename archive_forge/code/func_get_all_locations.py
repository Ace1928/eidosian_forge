import calendar
import time
import eventlet
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_log import versionutils
from oslo_utils import encodeutils
from glance.common import crypt
from glance.common import exception
from glance.common import timeutils
from glance import context
import glance.db as db_api
from glance.i18n import _, _LC, _LE, _LI, _LW
def get_all_locations(self):
    """Returns a list of image id and location tuple from scrub queue.

        :returns: a list of image id, location id and uri tuple from
            scrub queue

        """
    ret = []
    for image in self._get_all_images():
        deleted_at = image.get('deleted_at')
        if not deleted_at:
            continue
        deleted_at = timeutils.isotime(deleted_at)
        date_str = deleted_at.rsplit('.', 1)[0].rsplit(',', 1)[0]
        delete_time = calendar.timegm(time.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ'))
        if delete_time + self.scrub_time > time.time():
            continue
        for loc in image['locations']:
            if loc['status'] != 'pending_delete':
                continue
            if self.metadata_encryption_key:
                uri = crypt.urlsafe_decrypt(self.metadata_encryption_key, loc['url'])
            else:
                uri = loc['url']
            backend = loc['metadata'].get('store')
            if CONF.enabled_backends:
                ret.append((image['id'], loc['id'], uri, backend))
            else:
                ret.append((image['id'], loc['id'], uri))
    return ret