import copy
import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def get_size_from_uri_and_backend(uri, backend, context=None):
    """Retrieves image size from backend specified by uri."""
    loc = location.get_location_from_uri_and_backend(uri, backend, conf=CONF)
    store = get_store_from_store_identifier(backend)
    return store.get_size(loc, context=context)