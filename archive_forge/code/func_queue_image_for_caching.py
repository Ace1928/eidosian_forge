import os
from oslo_serialization import jsonutils as json
from glance.common import client as base_client
from glance.common import exception
from glance.i18n import _
def queue_image_for_caching(self, image_id):
    """
        Queue an image for prefetching into cache
        """
    self.do_request('PUT', '/queued_images/%s' % image_id)
    return True