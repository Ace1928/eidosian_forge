import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def get_store_from_uri(uri):
    """
    Given a URI, return the store object that would handle
    operations on the URI.

    :param uri: URI to analyze
    """
    scheme = uri[0:uri.find('/') - 1]
    return get_store_from_scheme(scheme)