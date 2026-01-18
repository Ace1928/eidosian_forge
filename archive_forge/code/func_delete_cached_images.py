import queue
import threading
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
from glance import image_cache
import glance.notifier
def delete_cached_images(self, req):
    """
        DELETE /cached_images - Clear all active cached images

        Removes all images from the cache.
        """
    self._enforce(req)
    return dict(num_deleted=self.cache.delete_all_cached_images())