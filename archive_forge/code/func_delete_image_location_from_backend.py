from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
def delete_image_location_from_backend(self, context, image_id, location):
    if CONF.delayed_delete:
        self.schedule_delayed_delete_from_backend(context, image_id, location)
    else:
        self.safe_delete_from_backend(context, image_id, location)