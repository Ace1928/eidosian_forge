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
def image_exists_in_cache(self, image_id):
    queued_images = self.cache.get_queued_images()
    if image_id in queued_images:
        return True
    cached_images = self.cache.get_cached_images()
    if image_id in [image['image_id'] for image in cached_images]:
        return True
    return False