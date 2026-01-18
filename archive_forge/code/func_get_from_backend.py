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
def get_from_backend(self, location, offset=0, chunk_size=None, context=None):
    try:
        scheme = location[:location.find('/') - 1]
        if scheme == 'unknown':
            raise store.UnknownScheme(scheme=scheme)
        return self.data[location]
    except KeyError:
        raise store.NotFound(image=location)