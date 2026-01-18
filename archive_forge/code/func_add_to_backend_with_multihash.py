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
def add_to_backend_with_multihash(self, conf, image_id, data, size, hashing_algo, scheme=None, context=None, verifier=None):
    for chunk in data:
        pass
    return super(FakeStoreAPIReader, self).add_to_backend_with_multihash(conf, image_id, data, size, hashing_algo, scheme=scheme, context=context, verifier=verifier)