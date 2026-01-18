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
def add_to_backend(self, conf, image_id, data, size, scheme=None, context=None, verifier=None):
    store_max_size = 7
    current_store_size = 2
    for location in self.data.keys():
        if image_id in location:
            raise exception.Duplicate()
    if not size:
        size = len(data.data.fd)
    if current_store_size + size > store_max_size:
        raise exception.StorageFull()
    if context.user_id == USER2:
        raise exception.Forbidden()
    if context.user_id == USER3:
        raise exception.StorageWriteDenied()
    self.data[image_id] = (data, size)
    checksum = 'Z'
    return (image_id, size, checksum, self.store_metadata)