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
def has_image(self, image_id):
    """Returns whether the queue contains an image or not.

        :param image_id: The opaque image identifier

        :returns: a boolean value to inform including or not
        """
    try:
        image = db_api.get_api().image_get(self.admin_context, image_id)
        return image['status'] == 'pending_delete'
    except exception.NotFound:
        return False