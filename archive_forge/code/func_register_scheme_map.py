import logging
import urllib.parse
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _
def register_scheme_map(scheme_map):
    """
    Given a mapping of 'scheme' to store_name, adds the mapping to the
    known list of schemes.

    This function overrides existing stores.
    """
    for k, v in scheme_map.items():
        LOG.debug('Registering scheme %s with %s', k, v)
        SCHEME_TO_CLS_MAP[k] = v